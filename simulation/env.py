#can left, right len 4
import os
import sys
import math
import traci
import traci.constants as tc
import time
import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces
# Frenet planner removed — dùng SUMO native setSpeed + changeLane

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

class SumoEnv(gym.Env):
    def __init__(self, render: bool = True, map_config = ["maps/TestMap/osm.sumocfg"],
              VTYPE_ID = "custom_passenger_car", TRAFFIC_SCALE = 0.8,
              test_mode: bool = False, test_route = "TestMap/test_route.rou.xml",
              imperfection = 0.0, impatience = 0.0, delay = 0) -> None:
        super().__init__()

        self.VEH_ID = "my_ego_car"
        self.VTYPE_ID = VTYPE_ID
        self.TRAFFIC_SCALE = TRAFFIC_SCALE
        self.render_mode: bool = render
        self.test_mode = test_mode
        self.test_route = test_route
        self.delay = delay
        self.step_count = 0
        self.MAX_EPISODE_STEPS = 1201

        # --- CONSTANTS ---
        self.MAX_SPEED = 55.6
        self.MAX_ACCEL = 4.15
        self.MAX_DECEL = 6.0
        self.MAX_ELEC = 120
        self.MAX_SLOPE = 20
        self.MAX_DIST = 100
        self.TARGET_SPEED_RATIO = 1.0   # bám 90% tốc độ giới hạn
        self.MIN_DESIRED_SPEED  = 5.0   # m/s — dưới mức này bị phạt too_slow
        self.JUNCTION_APPROACH_DIST = 15.0  # m — khoảng cách kích hoạt nhận diện xe gần ngã tư
        self.JUNCTION_TARGET_SPEED  = 2.0   # m/s — tốc độ mong muốn tại giao lộ
        self.JUNCTION_MIN_SPEED     = 0.0   # m/s — tốc độ tối thiểu (không phạt dưới mức này)
        self.JUNCTION_CONTEXT_RANGE = 100.0  # m — phạm vi context subscription tìm xe gần giao lộ
        self.RSS_DELTA = 0.5                  # s — Thời gian phản ứng
        self.RSS_D_MIN = 2.5                  # m — Khoảng cách an toàn tối thiểu khi dừng hẳn

        # --- ADAPTIVE REWARD ---
        # Lagrangian: mặc định TẮT để giữ baseline (reward gốc không đổi).
        self.adaptive_reward_enabled = False
        # Curriculum weighting theo episode — mặc định TẮT.
        self.curriculum_enabled = False
        self.curriculum_warmup = 1000
        self.curriculum_energy_w_start = 0.0
        self.curriculum_energy_w_end = 0.01197655138923567
        self.current_episode = 0

        self.maps = [map_config] if isinstance(map_config, str) else map_config
        self.imperfection = imperfection
        self.impatience = impatience

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # 45 chiều: 6 ego + 2 leader + 3 infra + 2 lane availability + 32 surroundings (8 xe × 4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(45,), dtype=np.float32)

        self.last_known_dist = 0.0
        self.LANE_CHANGE_THRESHOLD = 0.3   # |action[0]| > ngưỡng này thì đổi làn
        self.LANE_CHANGE_DURATION = 5.0    # giây giữ yêu cầu đổi làn

        # --- CACHE STORAGE ---
        self.veh_data = {}
        # OPT 1: Cache turn_info — vẫn cần cho step() để kích hoạt sumo_rescue_active
        self._turn_info_cache = (0.0, 1.0, 0.0)
        # OPT 2: Cache độ dài các edge trong route (static, không đổi trong episode)
        self._route_edge_lengths: dict = {}
        self._context_subscribed = False
        self._current_junction_tls_id = None
        self._was_left_turn_before_junction = False
        self._prev_was_in_junction = False

    def _veh_exists(self):
        try:
            return self.VEH_ID in traci.vehicle.getIDList()
        except Exception:
            return False

    def _update_cache(self):
        """Gọi API Traci 1 lần và lưu tất cả thông tin cần thiết vào biến self.veh_data"""
        if not self._veh_exists():
            self.veh_data = None
            return
        try:
            self.veh_data = {
                "speed":      traci.vehicle.getSpeed(self.VEH_ID),
                "accel":      traci.vehicle.getAcceleration(self.VEH_ID),
                "elec":       traci.vehicle.getElectricityConsumption(self.VEH_ID),
                "lane_idx":   traci.vehicle.getLaneIndex(self.VEH_ID),
                "lane_id":    traci.vehicle.getLaneID(self.VEH_ID),
                "road_id":    traci.vehicle.getRoadID(self.VEH_ID),
                "slope":      traci.vehicle.getSlope(self.VEH_ID),
                "lat_offset": traci.vehicle.getLateralLanePosition(self.VEH_ID),
                "lane_pos":   traci.vehicle.getLanePosition(self.VEH_ID),
                "leader":     traci.vehicle.getLeader(self.VEH_ID, dist=self.MAX_DIST),
                "tls":        traci.vehicle.getNextTLS(self.VEH_ID),
                "can_left":   4.0 if traci.vehicle.couldChangeLane(self.VEH_ID, 1) else 0.0,
                "can_right":  4.0 if traci.vehicle.couldChangeLane(self.VEH_ID, -1) else 0.0,
            }
            # OPT 3: Cập nhật turn_info cho step() (sumo_rescue_active), không dùng trong obs nữa
            self._turn_info_cache = self._get_next_turn_info()

            # Cache trạng thái rẽ trái trước khi vào junction
            if not self.veh_data["road_id"].startswith(":"):
                self._was_left_turn_before_junction = self._is_left_turn_at_junction()

            # Cache TLS ID khi đang trên đường (chưa vào junction)
            # Khi ego vào junction, getNextTLS trả về TLS của ngã tư TIẾP THEO → cần dùng cache
            if not self.veh_data["road_id"].startswith(":"):
                tls = self.veh_data["tls"]
                if tls and tls[0][2] < self.JUNCTION_CONTEXT_RANGE:
                    self._current_junction_tls_id = tls[0][0]
                else:
                    self._current_junction_tls_id = None
        except Exception:
            self.veh_data = None

    def _get_passenger_edges(self) -> list:
        all_edges = traci.edge.getIDList()
        valid_edges = []
        for edge_id in all_edges:
            if edge_id.startswith(":") or edge_id.startswith("!"):
                continue
            lane_id = f"{edge_id}_0"
            try:
                allowed = traci.lane.getAllowed(lane_id)
                if not allowed or "passenger" in allowed:
                    valid_edges.append(edge_id)
            except:
                continue
        return valid_edges

    # ------------------------------------------------------------------ #
    #  ROUTE-AWARE LANE GUIDANCE — chỉ dùng nội bộ cho sumo_rescue_active
    # ------------------------------------------------------------------ #
    def _get_turn_direction_numeric(self, from_edge: str, to_edge: str) -> float:
        try:
            def edge_heading(edge_id):
                shape = traci.edge.getShape(edge_id)
                if len(shape) < 2: return None
                x1, y1 = shape[0]; x2, y2 = shape[-1]
                return math.degrees(math.atan2(y2 - y1, x2 - x1))

            a1 = edge_heading(from_edge)
            a2 = edge_heading(to_edge)
            if a1 is None or a2 is None: return 0.0
            diff = (a2 - a1 + 360) % 360
            if diff > 180: diff -= 360
            return float(np.clip(-diff / 180.0, -1.0, 1.0))
        except Exception:
            return 0.0

    def _get_next_turn_info(self) -> tuple:
        """
        Trả về (turn_dir, turn_dist_norm, lane_offset) — chỉ dùng trong step()
        để tính sumo_rescue_active (turn_dist_n <= 0.3), không đưa vào obs nữa.
        """
        default = (0.0, 1.0, 0.0)
        if not self.veh_data: return default
        try:
            current_edge = self.veh_data["road_id"]
            if current_edge.startswith(":"): return default
            if not hasattr(self, "current_route_edges") or not self.current_route_edges: return default
            if current_edge not in self.current_route_edges: return default

            indices  = [i for i, x in enumerate(self.current_route_edges) if x == current_edge]
            curr_idx = indices[-1]
            if curr_idx >= len(self.current_route_edges) - 1: return default

            next_edge = self.current_route_edges[curr_idx + 1]
            turn_dir  = self._get_turn_direction_numeric(current_edge, next_edge)

            lane_id  = self.veh_data["lane_id"]
            if not lane_id: return (turn_dir, 1.0, 0.0)

            # OPT 4: Dùng cache độ dài làn thay vì gọi API mỗi bước
            if lane_id not in self._route_edge_lengths:
                self._route_edge_lengths[lane_id] = traci.lane.getLength(lane_id)
            lane_len = self._route_edge_lengths[lane_id]

            dist_left = max(0.0, lane_len - self.veh_data["lane_pos"])
            turn_dist_norm = float(np.clip(dist_left / max(lane_len, 1.0), 0.0, 1.0))

            num_lanes    = traci.edge.getLaneNumber(current_edge)
            correct_lane = None
            for li in range(num_lanes):
                try:
                    for link in traci.lane.getLinks(f"{current_edge}_{li}"):
                        if traci.lane.getEdgeID(link[0]) == next_edge:
                            correct_lane = li
                            break
                except Exception:
                    continue
                if correct_lane is not None: break

            if correct_lane is None: return (turn_dir, turn_dist_norm, 0.0)

            raw_offset  = correct_lane - self.veh_data["lane_idx"]
            lane_offset = float(np.clip(raw_offset / max(1, num_lanes - 1), -1.0, 1.0))
            return (turn_dir, turn_dist_norm, lane_offset)
        except Exception:
            return default

    # ------------------------------------------------------------------ #
    #  JUNCTION AWARENESS                                                 #
    # ------------------------------------------------------------------ #
    def _is_near_junction(self):
        """Kiểm tra xe ego có gần hoặc trong ngã tư không."""
        if not self.veh_data:
            return False
        road_id = self.veh_data["road_id"]
        # Đang trong junction (internal edge)
        if road_id.startswith(":"):
            return True
        # Gần ngã tư: kiểm tra khoảng cách tới TLS hoặc turn_info
        tls_data = self.veh_data["tls"]
        if tls_data and tls_data[0][2] < self.JUNCTION_APPROACH_DIST:
            return True
        # Kiểm tra thêm qua turn_info_cache (cho ngã tư không có đèn)
        _, turn_dist_n, _ = self._turn_info_cache
        if turn_dist_n < 1.0:
            try:
                lane_id = self.veh_data["lane_id"]
                if lane_id:
                    if lane_id not in self._route_edge_lengths:
                        self._route_edge_lengths[lane_id] = traci.lane.getLength(lane_id)
                    lane_len = self._route_edge_lengths[lane_id]
                    if turn_dist_n * lane_len < self.JUNCTION_APPROACH_DIST:
                        return True
            except Exception:
                pass
        return False

    # ------------------------------------------------------------------ #
    #  LEFT-TURN YIELDING DETECTION                                        #
    # ------------------------------------------------------------------ #
    def _is_left_turn_at_junction(self):
        """Kiểm tra xem hướng rẽ tiếp theo có phải rẽ trái không.
        Sử dụng SUMO link direction (link[6]): 'l'=left, 'L'=part-left, 't'=turn-around."""
        if not self.veh_data:
            return False

        road_id = self.veh_data["road_id"]

        # Đang trong junction → dùng giá trị đã cache từ lúc tiếp cận
        if road_id.startswith(":"):
            return getattr(self, '_was_left_turn_before_junction', False)

        # Kiểm tra route
        if not hasattr(self, 'current_route_edges') or not self.current_route_edges:
            return False

        current_edge = road_id
        if current_edge not in self.current_route_edges:
            return False

        indices = [i for i, x in enumerate(self.current_route_edges) if x == current_edge]
        curr_idx = indices[-1]
        if curr_idx >= len(self.current_route_edges) - 1:
            return False

        next_edge = self.current_route_edges[curr_idx + 1]

        # Kiểm tra SUMO link direction từ current edge tới next edge
        try:
            num_lanes = traci.edge.getLaneNumber(current_edge)
            for lane_idx in range(num_lanes):
                try:
                    links = traci.lane.getLinks(f"{current_edge}_{lane_idx}")
                except:
                    continue
                for link in links:
                    try:
                        link_next_edge = traci.lane.getEdgeID(link[0])
                    except:
                        continue
                    if link_next_edge == next_edge:
                        direction = link[6] if len(link) > 6 else 's'
                        # SUMO: 'l' = left, 'L' = partially left, 't' = turn around
                        if direction in ('l', 'L', 't'):
                            return True
        except:
            pass

        return False

    def _has_oncoming_traffic(self):
        """Kiểm tra có xe đang tiến về phía ego không.
        Dùng dot product giữa hướng đi của xe (velocity direction) và vector từ xe đến ego:
        - Dot > 0: xe đang tiến về phía ego (approaching)
        - Dot < 0: xe đang rời xa ego (receding)
        Chỉ tính xe KHÔNG nằm trên route của ego (xe đi thẳng, không cùng hướng)."""
        if not self.veh_data:
            return False

        try:
            ego_pos = traci.vehicle.getPosition(self.VEH_ID)

            context = traci.vehicle.getContextSubscriptionResults(self.VEH_ID)
            if not context:
                return False

            for veh_id, veh_data in context.items():
                if veh_id == self.VEH_ID:
                    continue

                veh_speed = veh_data.get(tc.VAR_SPEED, 0.0)
                # Xe đứng yên không phải mối đe dọa
                if veh_speed < 0.5:
                    continue

                veh_pos = veh_data.get(tc.VAR_POSITION, (0, 0))
                veh_angle = veh_data.get(tc.VAR_ANGLE, 0.0)

                # Vector từ xe đến ego
                to_ego_x = ego_pos[0] - veh_pos[0]
                to_ego_y = ego_pos[1] - veh_pos[1]
                dist = math.hypot(to_ego_x, to_ego_y)

                if dist < 1.0 or dist > 40.0:
                    continue

                # Hướng đi thực tế của xe (velocity direction)
                veh_rad = math.radians(90.0 - veh_angle)
                veh_hx = math.cos(veh_rad)
                veh_hy = math.sin(veh_rad)

                # Dot product: dương = xe đang tiến về phía ego
                dot = veh_hx * to_ego_x + veh_hy * to_ego_y

                if dot > 0:
                    # Xe đang tiến về phía ego
                    # Bỏ qua xe trên cùng route (xe đi cùng hướng với ego)
                    try:
                        veh_lane = veh_data.get(tc.VAR_LANE_ID, "")
                        veh_road = traci.lane.getEdgeID(veh_lane)
                        on_ego_route = (hasattr(self, 'current_route_edges')
                                        and veh_road in self.current_route_edges)
                        if not on_ego_route:
                            return True
                    except:
                        return True  # Không xác định được → coi như xe đối diện

            return False
        except:
            return False

    def _is_left_turn_yielding(self):
        """Kiểm tra xe có đang ở tình huống rẽ trái cần nhường đường không.
        Điều kiện kép: (1) hướng rẽ là trái (SUMO link direction) +
                       (2) gần/trong ngã tư +
                       (3) có xe đang tiến về phía ego (dot product)."""
        if not self._is_near_junction():
            return False
        if not self._is_left_turn_at_junction():
            return False
        return self._has_oncoming_traffic()

    def _get_cross_traffic_red_lanes(self):
        """Trả về set lane_id đang có đèn đỏ/vàng tại ngã tư gần nhất.
        Dùng để lọc bỏ xe đang dừng đèn đỏ khỏi context subscription.
        Khi ego ở TRONG junction, dùng TLS ID đã cache từ lúc tiếp cận."""
        red_lanes = set()
        if not self.veh_data:
            return red_lanes

        tls_id = None
        road_id = self.veh_data["road_id"]

        if road_id.startswith(":"):
            # Đang TRONG junction → getNextTLS trả về TLS ngã tư TIẾP THEO
            # → dùng TLS ID đã cache từ lúc tiếp cận ngã tư này
            tls_id = self._current_junction_tls_id
        else:
            # Đang trên đường → dùng TLS phía trước
            tls_data = self.veh_data["tls"]
            if tls_data:
                tls_id = tls_data[0][0]

        if not tls_id:
            return red_lanes

        try:
            state = traci.trafficlight.getRedYellowGreenState(tls_id)
            controlled_links = traci.trafficlight.getControlledLinks(tls_id)
            for i, link_group in enumerate(controlled_links):
                if i < len(state) and state[i].lower() in ('r', 'y'):
                    for link in link_group:
                        # link = (incoming_lane, outgoing_lane, via_lane)
                        if len(link) >= 1:
                            red_lanes.add(link[0])  # incoming lane đang đèn đỏ
            return red_lanes
        except Exception:
            return red_lanes

    def _calc_local_dx_dy(self, ego_pos, ego_angle_sumo, target_pos):
        """Tính toán tọa độ Descartes đối chuẩn từ tâm Ego.
        Output: dx_local (hướng tiến tới, + là trước), dy_local (hướng ngang, + là bên trái)"""
        dx_g = target_pos[0] - ego_pos[0]
        dy_g = target_pos[1] - ego_pos[1]
        
        # Chuyển góc SUMO (0=North, kim ĐH) sang Radian hệ Descartes (0=East, ngược kim)
        ego_rad = math.radians(90.0 - ego_angle_sumo)
        
        # Phép xoay Vector 2D sang Local Frame
        dx_local = dx_g * math.cos(ego_rad) + dy_g * math.sin(ego_rad)
        dy_local = -dx_g * math.sin(ego_rad) + dy_g * math.cos(ego_rad)
        
        return dx_local, dy_local

    def _get_surroundings(self):
        """Lấy thông tin xe xung quanh — hệ thống hợp nhất.
        - Gần/trong ngã tư: dùng context subscription, lọc xe đèn đỏ qua TLS,
          chỉ nhận xe đang trong junction hoặc sắp vào (loại bỏ xe đã qua).
          Lấy 8 xe gần nhất.
        - Đường thẳng: dùng getNeighbors cho 4 hướng (LF, LB, RF, RB) và Follower.
        Trả về 32 giá trị: 8 bộ (dx_norm, dy_norm, rel_speed, rel_angle_norm)"""
        my_speed = self.veh_data["speed"] if self.veh_data else 0.0
        max_n = 8
        # Default: xe cực xa trước mặt (dx_norm=1.0, dy_norm=0.0)
        result = [1.0, 0.0, 0.0, 0.0] * max_n  # 32 giá trị

        recognized_veh_ids = set()
        near_junction = self._is_near_junction()

        if near_junction:
            # --- CONTEXT SUBSCRIPTION MODE ---
            try:
                # Subscribe nếu chưa
                if not self._context_subscribed:
                    traci.vehicle.subscribeContext(
                        self.VEH_ID,
                        tc.CMD_GET_VEHICLE_VARIABLE,
                        self.JUNCTION_CONTEXT_RANGE,
                        [tc.VAR_SPEED, tc.VAR_POSITION, tc.VAR_LANE_ID, tc.VAR_ANGLE]
                    )
                    self._context_subscribed = True

                context = traci.vehicle.getContextSubscriptionResults(self.VEH_ID)
                if not context:
                    return result, recognized_veh_ids

                # Lấy danh sách lane đèn đỏ để lọc (chính xác qua TLS state)
                red_lanes = self._get_cross_traffic_red_lanes()

                # Vị trí và heading ego (lấy on-demand)
                ego_pos = traci.vehicle.getPosition(self.VEH_ID)
                ego_angle = traci.vehicle.getAngle(self.VEH_ID)  # SUMO: 0=North, clockwise
                ego_road = self.veh_data["road_id"]
                ego_in_junction = ego_road.startswith(":")

                candidates = []
                for veh_id, veh_data in context.items():
                    if veh_id == self.VEH_ID:
                        continue

                    veh_lane = veh_data.get(tc.VAR_LANE_ID, "")
                    veh_speed = veh_data.get(tc.VAR_SPEED, 0.0)
                    veh_pos = veh_data.get(tc.VAR_POSITION, (0, 0))
                    veh_angle = veh_data.get(tc.VAR_ANGLE, 0.0)

                    # Xác định road_id từ lane_id
                    try:
                        veh_road = traci.lane.getEdgeID(veh_lane)
                    except:
                        veh_road = veh_lane.rsplit("_", 1)[0] if "_" in veh_lane else veh_lane

                    # --- LỌC XE ĐÈN ĐỎ: lọc theo lane đang đèn đỏ thực tế ---
                    if veh_lane in red_lanes:
                        continue

                    # --- LỌC XE ĐÃ QUA NGÃ TƯ ---
                    # Chỉ giữ xe: (1) đang TRONG junction, hoặc (2) ĐANG TIẾN VÀO junction
                    # Bỏ xe đã thoát khỏi junction (trên departing edge)
                    veh_in_junction = veh_road.startswith(":")
                    dx_local, dy_local = self._calc_local_dx_dy(ego_pos, ego_angle, veh_pos)
                    
                    if not veh_in_junction:
                        if ego_in_junction and dx_local < 0:
                            continue

                        # --- LỌC XE TRÊN LÀN NGƯỢC CHIỀU ---
                        # Bỏ qua xe trên làn đang rời khỏi ngã tư (không phải mối đe dọa)
                        # Giữ lại xe trên route của ego (cùng hướng đi)
                        on_ego_route = (hasattr(self, 'current_route_edges')
                                        and veh_road in self.current_route_edges)
                        if not on_ego_route:
                            # Tính khoảng cách xe đến lối vào ngã tư
                            try:
                                veh_lane_pos = traci.vehicle.getLanePosition(veh_id)
                                veh_lane_len = traci.lane.getLength(veh_lane)
                                dist_to_junc = veh_lane_len - veh_lane_pos
                            except:
                                dist_to_junc = float('inf')

                            # Xe ≤ 10m trước ngã tư → LUÔN giữ lại (đang tiến vào, mối đe dọa)
                            # Xe > 10m → kiểm tra heading: chỉ giữ xe đang hướng về phía ego
                            if dist_to_junc > 10.0:
                                veh_rad = math.radians(90.0 - veh_angle)
                                veh_hx = math.cos(veh_rad)
                                veh_hy = math.sin(veh_rad)
                                to_ego_x = ego_pos[0] - veh_pos[0]
                                to_ego_y = ego_pos[1] - veh_pos[1]
                                dot = veh_hx * to_ego_x + veh_hy * to_ego_y
                                if dot < 0:
                                    continue  # Xe xa ngã tư VÀ đang rời xa → bỏ qua

                    dist = math.hypot(dx_local, dy_local)
                    if dist < 0.1:  # quá gần, có thể trùng vị trí
                        continue

                    # Góc heading tương đối
                    rel_heading = (veh_angle - ego_angle + 360) % 360
                    if rel_heading > 180:
                        rel_heading -= 360
                    rel_angle_norm = rel_heading / 180.0  # [-1, 1]

                    dx_norm = float(np.clip(dx_local / self.MAX_DIST, -1.0, 1.0))
                    dy_norm = float(np.clip(dy_local / self.MAX_DIST, -1.0, 1.0))

                    candidates.append((dist, veh_speed, rel_angle_norm, dx_norm, dy_norm, veh_id))

                # Sắp xếp theo khoảng cách Euclidean, lấy 8 xe gần nhất để truyền sang Observation
                candidates.sort(key=lambda x: x[0])
                for i, (dist, v_speed, angle, dx_n, dy_n, v_id) in enumerate(candidates[:max_n]):
                    result[i * 4]     = dx_n
                    result[i * 4 + 1] = dy_n
                    result[i * 4 + 2] = (my_speed - v_speed) / self.MAX_SPEED
                    result[i * 4 + 3] = angle
                    recognized_veh_ids.add(v_id)

            except Exception:
                pass
        else:
            # --- NORMAL MODE: getNeighbors (đường thẳng) ---
            # Hủy context subscription nếu đang active
            if self._context_subscribed:
                try:
                    traci.vehicle.unsubscribeContext(
                        self.VEH_ID, tc.CMD_GET_VEHICLE_VARIABLE, self.JUNCTION_CONTEXT_RANGE
                    )
                except Exception:
                    pass
                self._context_subscribed = False

            try:
                ego_pos = traci.vehicle.getPosition(self.VEH_ID)
                ego_angle = traci.vehicle.getAngle(self.VEH_ID)

                # LF=0,1,2,3  LB=4,5,6,7  RF=8,9,10,11  RB=12,13,14,15 (4 hướng × 4)
                def process_side(neighbors, f_idx, b_idx):
                    closest_f = float("inf")
                    closest_b = float("inf")
                    best_f_id = None
                    best_b_id = None
                    for n_id, dist in neighbors:
                        try:
                            n_speed = traci.vehicle.getSpeed(n_id)
                            n_angle = traci.vehicle.getAngle(n_id)
                            n_pos   = traci.vehicle.getPosition(n_id)
                        except:
                            continue
                        # Góc heading tương đối
                        rel_heading = (n_angle - ego_angle + 360) % 360
                        if rel_heading > 180:
                            rel_heading -= 360
                        angle = rel_heading / 180.0

                        dx_local, dy_local = self._calc_local_dx_dy(ego_pos, ego_angle, n_pos)
                        dx_norm = float(np.clip(dx_local / self.MAX_DIST, -1.0, 1.0))
                        dy_norm = float(np.clip(dy_local / self.MAX_DIST, -1.0, 1.0))

                        if dist > 0:
                            if dist < closest_f:
                                closest_f = dist
                                best_f_id = n_id
                                result[f_idx]     = dx_norm
                                result[f_idx + 1] = dy_norm
                                result[f_idx + 2] = (my_speed - n_speed) / self.MAX_SPEED
                                result[f_idx + 3] = angle
                        else:
                            adist = abs(dist)
                            if adist < closest_b:
                                closest_b = adist
                                best_b_id = n_id
                                result[b_idx]     = dx_norm
                                result[b_idx + 1] = dy_norm
                                result[b_idx + 2] = (my_speed - n_speed) / self.MAX_SPEED
                                result[b_idx + 3] = angle
                    if best_f_id: recognized_veh_ids.add(best_f_id)
                    if best_b_id: recognized_veh_ids.add(best_b_id)

                process_side(traci.vehicle.getNeighbors(self.VEH_ID, 2), 0, 4)   # trái
                process_side(traci.vehicle.getNeighbors(self.VEH_ID, 1), 8, 12)  # phải

                # Sửa điểm mù: Bổ sung xe bám đằng đuôi (Follower) trên cùng làn
                try:
                    follower = traci.vehicle.getFollower(self.VEH_ID)
                    if follower and follower[0]:
                        # Lưu vào slot [16:20]
                        process_side([(follower[0], -follower[1])], 16, 20)
                except:
                    pass
            except:
                pass

        return result, recognized_veh_ids

    def _get_obs(self):
        if self.veh_data is None:
            return np.zeros(45, dtype=np.float32)

        d = self.veh_data

        try:
            # 1. EGO PHYSICS (6 chiều)
            velocity     = np.clip(d["speed"] / self.MAX_SPEED, 0.0, 2.0)
            acceleration = np.clip(d["accel"] / self.MAX_ACCEL, -1.0, 1.0)
            elec         = np.clip(d["elec"]  / self.MAX_ELEC,  0.0, 5.0)

            lane_idx    = d["lane_idx"]
            road_id     = d["road_id"]
            total_lanes = traci.edge.getLaneNumber(road_id)
            norm_lane   = lane_idx / max(1, total_lanes - 1)

            slope      = np.clip(d["slope"]      / self.MAX_SLOPE, -1.0, 1.0)
            lat_offset = np.clip(d["lat_offset"], -10.0, 10.0)

            # 2. SURROUNDINGS (32 chiều: 8 xe × 4: dist, rel_speed, angle, bearing)
            surroundings, recognized_veh_ids = self._get_surroundings()

            # 3. LEADER (2 chiều) — luôn lấy
            leader = d["leader"]
            leader_id = None
            if leader:
                leader_id = leader[0]
                l_dist      = min(leader[1], self.MAX_DIST) / self.MAX_DIST
                l_rel_speed = (d["speed"] - traci.vehicle.getSpeed(leader_id)) / self.MAX_SPEED
            else:
                l_dist, l_rel_speed = 1.0, 0.0

            # --- COLORIZER (Đổi màu xe nhận diện) ---
            if not hasattr(self, "_last_recognized_vehicles"):
                self._last_recognized_vehicles = set()

            current_recognized = set(recognized_veh_ids)
            if leader_id:
                current_recognized.add(leader_id)

            # Khôi phục màu mặc định (vàng) cho xe không còn nằm trong obs
            for v_id in self._last_recognized_vehicles - current_recognized:
                try: traci.vehicle.setColor(v_id, (255, 255, 0))
                except: pass

            # Neighbors: màu đỏ
            for v_id in recognized_veh_ids:
                try: traci.vehicle.setColor(v_id, (255, 0, 0))
                except: pass

            # Leader: màu xanh lam
            if leader_id:
                try: traci.vehicle.setColor(leader_id, (0, 0, 255))
                except: pass

            self._last_recognized_vehicles = current_recognized

            # 4. INFRASTRUCTURE (3 chiều)
            lane_id     = d["lane_id"]
            speed_limit = traci.lane.getMaxSpeed(lane_id) / self.MAX_SPEED if lane_id else 1.0

            tls_data = d["tls"]
            if tls_data:
                tls_dist  = min(tls_data[0][2], self.MAX_DIST) / self.MAX_DIST
                tls_state = 1.0 if tls_data[0][3].lower() == 'g' else 0.0
            else:
                tls_dist, tls_state = 1.0, 1.0

            # 5. LANE AVAILABILITY (2 chiều)
            can_left  = d.get("can_left", 0.0)
            can_right = d.get("can_right", 0.0)

            obs_list = [
                # EGO PHYSICS        [0..5]
                velocity, acceleration, elec, norm_lane, slope, lat_offset,
                # LEADER             [6..7]
                l_dist, l_rel_speed,
                # INFRA              [8..10]
                speed_limit, tls_dist, tls_state,
                # LANE AVAIL         [11..12]
                can_left, can_right,
                # SURROUNDINGS       [13..44]
            ] + surroundings   # 32 chiều (8 xe × 4: dist, rel_speed, angle, bearing)

            obs = np.array(obs_list, dtype=np.float64)
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            obs = np.clip(obs, -5.0, 5.0)
            return obs.astype(np.float32)

        except Exception:
            return np.zeros(45, dtype=np.float32)

    def _get_dist_to_destination(self):
        try:
            if not self.veh_data: return 2100.0
            current_edge = self.veh_data["road_id"]

            if current_edge.startswith(":"):
                return self.last_known_dist if self.last_known_dist else 2100.0

            if hasattr(self, "current_route_edges") and current_edge in self.current_route_edges:
                indices = [i for i, x in enumerate(self.current_route_edges) if x == current_edge]
                idx = indices[-1]
                remaining_edges = self.current_route_edges[idx:]

                # OPT 7: Dùng cache độ dài edge thay vì gọi API traci mỗi step
                dist = 0.0
                for e in remaining_edges:
                    key = f"{e}_0"
                    if key not in self._route_edge_lengths:
                        self._route_edge_lengths[key] = traci.lane.getLength(key)
                    dist += self._route_edge_lengths[key]

                dist -= self.veh_data["lane_pos"]
                self.last_known_dist = dist
                return dist
            return 2100.0
        except:
            return 2100.0

    def _get_junction_proximity(self):
        """
        Tính khoảng cách đến giao lộ/nút giao gần nhất.
        Kết hợp 2 nguồn: (1) turn_info_cache từ route, (2) TLS distance.
        Trả về khoảng cách thực (meters) đến junction gần nhất, hoặc inf nếu không gần.
        """
        if not self.veh_data: return float('inf')
        
        # Nếu đang ở trong ngã tư, khoảng cách = 0
        if self.veh_data["road_id"].startswith(":"):
            return 0.0

        dist_to_junction = float('inf')

        # Nguồn 1: Khoảng cách đến điểm rẽ tiếp theo (từ route)
        _, turn_dist_n, _ = self._turn_info_cache
        if turn_dist_n < 1.0:  # đang trên edge có turn phía trước
            try:
                lane_id = self.veh_data["lane_id"]
                if lane_id:
                    if lane_id not in self._route_edge_lengths:
                        self._route_edge_lengths[lane_id] = traci.lane.getLength(lane_id)
                    lane_len = self._route_edge_lengths[lane_id]
                    dist_to_junction = min(dist_to_junction, turn_dist_n * lane_len)
            except Exception:
                pass

        # Nguồn 2: Khoảng cách đến đèn giao thông (TLS)
        tls_data = self.veh_data.get("tls")
        if tls_data:
            tls_dist_raw = tls_data[0][2]
            dist_to_junction = min(dist_to_junction, tls_dist_raw)

        return dist_to_junction

    def _calculate_reward(self, action):
        if not self.veh_data: return 0.0
        d = self.veh_data

        # ------------------------------------------------------------------ #
        #  TRỌNG SỐ                                                           #
        # ------------------------------------------------------------------ #
        W_SPEED_TARGET  =  0.3
        W_TOO_SLOW      = -0.1
        W_PROGRESS      =  0.18692291992838891
        W_ENERGY        = -0.01197655138923567
        W_COMFORT       = -0.0287413370864677
        W_LANE_CHANGE   = -0.1
        W_SAFETY        = -0.019335217679737792
        W_JUNCTION      = -0.25850726243996763

        W_RED_LIGHT     = -25.0
        W_TIME          = -0.04447192165875163

        cur_speed = d["speed"]
        lane_id   = d["lane_id"]
        road_id   = d["road_id"]

        # ------------------------------------------------------------------ #
        #  1. SPEED — bell-curve theo tốc độ giới hạn (từ wrappers.py)       #
        #  Thay thế linear cũ: không còn phạt cứng khi < 3 m/s               #
        # ------------------------------------------------------------------ #
        try:
            speed_limit_ms = traci.lane.getMaxSpeed(lane_id) if lane_id else self.MAX_SPEED
        except:
            speed_limit_ms = self.MAX_SPEED

        # Xác định đèn xanh và khoảng cách tới ngã tư
        dist_to_junc = self._get_junction_proximity()
        in_junction = road_id.startswith(":")
        
        is_green_light = False
        tls_data = d.get("tls")
        if tls_data and len(tls_data) > 0 and tls_data[0][2] < self.JUNCTION_APPROACH_DIST:
            if tls_data[0][3].lower() in ('g', 'G'):
                is_green_light = True

        # Target speed linh hoạt
        base_target = max(speed_limit_ms * self.TARGET_SPEED_RATIO, self.MIN_DESIRED_SPEED)
        if in_junction:
            # Rẽ trái trong ngã tư: target = 0 (cho phép dừng hẳn nhường đường)
            if getattr(self, '_was_left_turn_before_junction', False):
                target_speed = 0.0
            else:
                target_speed = self.JUNCTION_TARGET_SPEED
        elif dist_to_junc < self.JUNCTION_APPROACH_DIST and not is_green_light:
            # Gần ngã tư (không phải đèn xanh), giảm tốc độ dần
            alpha = dist_to_junc / self.JUNCTION_APPROACH_DIST
            target_speed = alpha * base_target + (1 - alpha) * self.JUNCTION_TARGET_SPEED
        else:
            # Ngoài ngã tư hoặc đang đèn xanh, bám tốc độ max
            target_speed = base_target

        speed_error  = (cur_speed - target_speed) / max(target_speed, 1.0)
        r_speed_target = float(np.exp(-3.0 * speed_error ** 2))  # [0, 1], đỉnh tại target

        # Too-slow penalty: tắt khi gần hoặc trong giao lộ
        if dist_to_junc < self.JUNCTION_APPROACH_DIST or in_junction:
            r_too_slow = 0.0  # gần/trong giao lộ → không phạt chạy chậm
        elif cur_speed < self.MIN_DESIRED_SPEED:
            r_too_slow = 2.0 * (1.0 - cur_speed / self.MIN_DESIRED_SPEED)
        else:
            r_too_slow = 0.0

        # ------------------------------------------------------------------ #
        #  2. PROGRESS — tiến về đích                                        #
        # ------------------------------------------------------------------ #
        dist = self._get_dist_to_destination()
        if not hasattr(self, "prev_dist"): self.prev_dist = dist
        progress_reward = float(np.clip(self.prev_dist - dist, -1.0, 1.0))
        self.prev_dist = dist

        # ------------------------------------------------------------------ #
        #  3. ENERGY EFFICIENCY — Wh/m thay vì Wh/s (từ wrappers.py)        #
        #  Ngăn speed-collapse: xe chậm tốn Wh/m cao hơn xe nhanh            #
        # ------------------------------------------------------------------ #
        elec = max(0.0, d["elec"])
        if cur_speed > 0.5:
            wh_per_meter   = elec / cur_speed
            energy_penalty = float(np.clip(wh_per_meter / 0.5, 0.0, 1.0))
        else:
            energy_penalty = 1.0  # dừng hẳn → hiệu suất tệ nhất

        # ------------------------------------------------------------------ #
        #  4. COMFORT — jerk ga/phanh                                        #
        # ------------------------------------------------------------------ #
        if not hasattr(self, "prev_action"): self.prev_action = action
        accel_jerk = float(np.abs(action[1] - self.prev_action[1]))
        self.prev_action = action
        self.last_jerk = accel_jerk

        # ------------------------------------------------------------------ #
        #  5. SAFETY — TTC-based (từ wrappers.py)                            #
        #  safe_dist = 1.5s headway + 5m gap                                 #
        #  Thay thế exp-penalty cũ dùng khoảng cách cố định                  #
        # ------------------------------------------------------------------ #
        leader = d["leader"]
        safe_dist = cur_speed * 1.5 + 5.0
        if leader is not None:
            leader_dist_m = leader[1]
            if leader_dist_m < safe_dist:
                safety_penalty = float(np.clip(
                    (safe_dist - leader_dist_m) / safe_dist, 0.0, 1.0
                ))
            else:
                safety_penalty = 0.0
        else:
            safety_penalty = 0.0

        # ------------------------------------------------------------------ #
        #  6. RED LIGHT — phạt 1 lần khi xe vượt qua vạch dừng đèn đỏ/vàng #
        # ------------------------------------------------------------------ #
        road_id = d["road_id"]
        red_light_penalty = 0.0
        if road_id.startswith(":") and getattr(self, "_prev_tls_was_red", False):
            red_light_penalty = 1.0   # W_RED_LIGHT × 1.0 = -5.0 / lần vượt

        tls_data = d["tls"]
        if tls_data:
            tls_dist_raw  = tls_data[0][2]
            tls_state_str = tls_data[0][3].lower()
            self._prev_tls_was_red = (
                ('r' in tls_state_str or 'y' in tls_state_str)
                and tls_dist_raw < 3.0
            )
        else:
            self._prev_tls_was_red = False

        # ------------------------------------------------------------------ #
        #  7. JUNCTION APPROACH — phạt chạy nhanh gần giao lộ                #
        #  Chỉ phạt khi speed > JUNCTION_TARGET_SPEED và gần junction.       #
        #  Không phạt khi speed <= JUNCTION_MIN_SPEED (0.5 m/s).             #
        #  Proximity factor: tăng dần khi càng gần (1.0 tại junction,        #
        #                    0.0 tại JUNCTION_APPROACH_DIST).                #
        #  Speed excess: bình phương → phạt nặng khi chạy quá nhanh.         #
        #  Anti-exploitation: W_PROGRESS + W_TOO_SLOW + W_TIME + Wh/m        #
        #    đảm bảo xe không dừng/crawl để né phạt junction.                #
        # ------------------------------------------------------------------ #
        junction_penalty = 0.0

        if dist_to_junc < self.JUNCTION_APPROACH_DIST:
            # Nếu đang tiếp cận đèn xanh mà MỚI CHỈ Ở NGOÀI ngã tư, miễn phạt tốc độ lố
            if not in_junction and is_green_light:
                junction_penalty = 0.0
            else:
                proximity = 1.0 - (dist_to_junc / self.JUNCTION_APPROACH_DIST)
                effective_speed = max(cur_speed, self.JUNCTION_MIN_SPEED)
                speed_excess = max(0.0, effective_speed - self.JUNCTION_TARGET_SPEED)

                if speed_excess > 0.0:
                    max_excess = max(speed_limit_ms - self.JUNCTION_TARGET_SPEED, 1.0)
                    norm_excess = speed_excess / max_excess  # [0, 1+]
                    junction_penalty = float(proximity * (norm_excess ** 2))
                    junction_penalty = min(junction_penalty, 1.0)  # cap tại 1.0

        # ------------------------------------------------------------------ #
        #  8. LANE CHANGE — phạt action[0] xa giá trị 0.0                    #
        #  Bình phương để phạt nhẹ khi gần 0, phạt nặng khi xa 0.            #
        # ------------------------------------------------------------------ #
        lane_change_penalty = float(action[0] ** 2)  # [0, 1], 0 khi giữ làn

        # ------------------------------------------------------------------ #
        #  TỔNG HỢP                                                          #
        # ------------------------------------------------------------------ #
        r_speed_target      = np.nan_to_num(r_speed_target)
        r_too_slow          = np.nan_to_num(r_too_slow)
        progress_reward     = np.nan_to_num(progress_reward)
        energy_penalty      = np.nan_to_num(energy_penalty)
        accel_jerk          = np.nan_to_num(accel_jerk)
        lane_change_penalty = np.nan_to_num(lane_change_penalty)
        safety_penalty      = np.nan_to_num(safety_penalty)
        red_light_penalty   = np.nan_to_num(red_light_penalty)
        junction_penalty    = np.nan_to_num(junction_penalty)

        # ------------------------------------------------------------------ #
        #  TỔNG HỢP — tách reward_task và các cost bị ràng buộc (Lagrangian) #
        # ------------------------------------------------------------------ #
        reward_task = (
            (r_speed_target      * W_SPEED_TARGET)  +
            (r_too_slow          * W_TOO_SLOW)      +
            (progress_reward     * W_PROGRESS)      +
            (energy_penalty      * W_ENERGY)        +
            (lane_change_penalty * W_LANE_CHANGE)   +
            (junction_penalty    * W_JUNCTION)      +
            W_TIME
        )

        # Cost ≥ 0 (sẽ bị trừ λ·cost ở tầng agent). Mỗi cost là đại lượng THÔ
        # (chưa nhân trọng số) để λ tương ứng tự học mức phạt.
        cost_safety   = float(safety_penalty)
        cost_comfort  = float(accel_jerk)
        cost_redlight = float(red_light_penalty)

        if getattr(self, "adaptive_reward_enabled", False):
            return float(reward_task), cost_safety, cost_comfort, cost_redlight

        # Backward-compatible: scalar y HỆT công thức gốc (9 thành phần + W_TIME)
        return (
            reward_task
            + (accel_jerk        * W_COMFORT)
            + (safety_penalty    * W_SAFETY)
            + (red_light_penalty * W_RED_LIGHT)
        )

    def _split_reward_for_step(self, reward_out):
        """Chuẩn hoá output của _calculate_reward về
        (reward_task, cost_safety, cost_comfort, cost_redlight).
        Khi adaptive tắt, reward_out là scalar → các cost = 0."""
        if isinstance(reward_out, tuple):
            reward_task, cost_safety, cost_comfort, cost_redlight = reward_out
            return (float(reward_task), float(cost_safety),
                    float(cost_comfort), float(cost_redlight))
        return float(reward_out), 0.0, 0.0, 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        try:
            traci.close()
        except Exception:
            pass
        time.sleep(0.5)

        if self.test_mode:
            active_map = self.maps[0]
            route_arg  = ["-a", self.test_route]
        else:
            active_map = random.choice(self.maps)
            route_arg  = []

        self.step_count = 0
        self.veh_data   = None
        # OPT 8: Xoá cache độ dài edge khi reset episode (map mới có thể khác)
        self._route_edge_lengths = {}
        self._turn_info_cache    = (0.0, 1.0, 0.0)
        self._prev_tls_was_red   = False
        self._context_subscribed = False
        self._current_junction_tls_id = None
        self._last_recognized_vehicles = set()
        self._was_left_turn_before_junction = False
        self._prev_was_in_junction = False

        SumoBinary = "sumo-gui" if self.render_mode else "sumo"
        SumoCMD = [SumoBinary, "-c", active_map] + route_arg + \
                ["--start", "--quit-on-end",
                "--device.emissions.probability", "1.0",
                "--scale", str(self.TRAFFIC_SCALE),
                "--delay", str(self.delay),
                "--no-step-log", "true",
                "--time-to-teleport", "-1",
                "--collision.action", "remove",
                "--collision.check-junctions", "true",
                "--no-warnings", "true"]

        traci.start(SumoCMD)
        self._success = False

        if hasattr(self, "prev_action"): del self.prev_action
        if hasattr(self, "prev_dist"):   del self.prev_dist
        self.last_jerk = 0.0

        # Warmup
        for _ in range(random.randint(300, 600)): traci.simulationStep()

        try:
            existing_types = traci.vehicletype.getIDList()
            source_type    = "DEFAULT_VEHTYPE"
            if source_type not in existing_types and len(existing_types) > 0:
                source_type = existing_types[0]

            traci.vehicletype.copy(source_type, self.VTYPE_ID)
            traci.vehicletype.setVehicleClass(self.VTYPE_ID, "passenger")
            traci.vehicletype.setColor(self.VTYPE_ID, (0, 255, 0))
            traci.vehicletype.setParameter(self.VTYPE_ID, "mass",   "2911")
            traci.vehicletype.setLength(self.VTYPE_ID, "5.1181")
            traci.vehicletype.setEmissionClass(self.VTYPE_ID, "MMPEVEM")

            traci.vehicletype.setParameter(self.VTYPE_ID, "has.battery.device",                    "true")
            traci.vehicletype.setParameter(self.VTYPE_ID, "device.battery.capacity",               "123000.00")
            traci.vehicletype.setParameter(self.VTYPE_ID, "device.battery.chargeLevel",            "123000.00")
            traci.vehicletype.setParameter(self.VTYPE_ID, "device.battery.rechargeEfficiency",     "0.8")
            traci.vehicletype.setParameter(self.VTYPE_ID, "device.battery.maxRegenerationAcceleration", "2.0")

            for v_type in existing_types:
                traci.vehicletype.setParameter(v_type, "sigma",      str(self.imperfection))
                traci.vehicletype.setParameter(v_type, "impatience", str(self.impatience))
        except Exception:
            pass

        spawned = False
        if self.test_mode:
            while self.VEH_ID not in traci.vehicle.getIDList():
                traci.simulationStep()
            traci.vehicle.setType(self.VEH_ID, self.VTYPE_ID)
            traci.vehicle.setSpeedMode(self.VEH_ID, 0)
            traci.vehicle.setLaneChangeMode(self.VEH_ID, 768)  # 256+512: cho phép đổi làn qua TraCI
            # Context subscription giờ là dynamic — subscribe/unsubscribe trong _get_surroundings()
            for _ in range(50):
                traci.simulationStep()
                if self.VEH_ID in traci.vehicle.getIDList():
                    spawned = True
                    break
        else:
            self.drivable_edges = self._get_passenger_edges()

            for attempt in range(20):
                if not self.drivable_edges: break
                start_edge   = random.choice(self.drivable_edges)
                route_edges  = [start_edge]
                visited_edges = {start_edge}   # OPT 9: set thay list → O(1) lookup
                current_len  = 0.0
                try:
                    current_len += traci.lane.getLength(f"{start_edge}_0")
                except:
                    continue

                curr_edge_id = start_edge
                dead_end     = False
                while current_len < 2100.0:
                    try:
                        num_lanes = traci.edge.getLaneNumber(curr_edge_id)
                    except:
                        dead_end = True
                        break

                    # OPT 10: Dùng dict để lưu edge và hướng rẽ thực tế (từ SUMO)
                    valid_next_dict = {}
                    for lane_idx in range(num_lanes):
                        try:
                            links = traci.lane.getLinks(f"{curr_edge_id}_{lane_idx}")
                        except:
                            continue
                        for link in links:
                            next_lane_id = link[0]
                            direction = link[6] if len(link) > 6 else 's'
                            try:
                                next_edge_id = traci.lane.getEdgeID(next_lane_id)
                            except:
                                continue
                            if (not next_edge_id.startswith(":")
                                    and next_edge_id in self.drivable_edges
                                    and next_edge_id not in visited_edges):
                                valid_next_dict[next_edge_id] = direction

                    if not valid_next_dict:
                        dead_end = True
                        break

                    left_edges = []
                    other_edges = []
                    for pedge, pdir in valid_next_dict.items():
                        # SUMO direction: 'l' = left, 'L' = part left, 't' = turn around
                        if pdir in ('l', 'L', 't'):
                            left_edges.append(pedge)
                        else:
                            other_edges.append(pedge)

                    if left_edges and other_edges:
                        # 95% xác suất chọn rẽ trái nếu có thể
                        if random.random() < 0.65:
                            next_edge = random.choice(left_edges)
                        else:
                            next_edge = random.choice(other_edges)
                    elif left_edges:
                        next_edge = random.choice(left_edges)
                    else:
                        next_edge = random.choice(other_edges)
                    route_edges.append(next_edge)
                    visited_edges.add(next_edge)
                    try: current_len += traci.lane.getLength(f"{next_edge}_0")
                    except: pass
                    curr_edge_id = next_edge

                if not dead_end and current_len >= 2100.0:
                    try:
                        route_id = f"route_{random.randint(0, 999999)}"
                        traci.route.add(route_id, route_edges)
                        self.current_route_edges = route_edges
                        traci.vehicle.add(self.VEH_ID, route_id, departPos="free", typeID=self.VTYPE_ID)
                        for _ in range(50):
                            traci.simulationStep()
                            if self.VEH_ID in traci.vehicle.getIDList():
                                spawned = True
                                break
                        if spawned:
                            traci.vehicle.setSpeedMode(self.VEH_ID, 0)
                            traci.vehicle.setLaneChangeMode(self.VEH_ID, 768)  # 256+512: cho phép đổi làn qua TraCI
                            # Context subscription giờ là dynamic — subscribe/unsubscribe trong _get_surroundings()
                            self.last_known_dist = current_len
                            self.prev_dist       = current_len
                            break
                        else:
                            try: traci.vehicle.remove(self.VEH_ID)
                            except: pass
                    except Exception:
                        try: traci.vehicle.remove(self.VEH_ID)
                        except: pass
                        continue

            if not spawned:
                return self.reset(seed=seed, options=options)

        if self.render_mode and spawned:
            if self.VEH_ID in traci.vehicle.getIDList():
                traci.gui.trackVehicle("View #0", self.VEH_ID)
                traci.gui.setZoom("View #0", 1001)

        self.stuck_time = 0
        self._update_cache()

        obs = self._get_obs()
        return obs, {}

    def _get_obstacles_for_planner(self):
        obstacles = []
        try:
            context = traci.vehicle.getContextSubscriptionResults(self.VEH_ID)
            if not context:
                traci.vehicle.subscribeContext(
                    self.VEH_ID,
                    tc.CMD_GET_VEHICLE_VARIABLE,
                    self.JUNCTION_CONTEXT_RANGE,
                    [tc.VAR_SPEED, tc.VAR_POSITION, tc.VAR_ANGLE]
                )
                context = traci.vehicle.getContextSubscriptionResults(self.VEH_ID)
            
            if context:
                for veh_id, veh_data in context.items():
                    if veh_id == self.VEH_ID:
                        continue
                    pos = veh_data.get(tc.VAR_POSITION, (0, 0))
                    speed = veh_data.get(tc.VAR_SPEED, 0.0)
                    angle = veh_data.get(tc.VAR_ANGLE, 0.0)
                    obstacles.append({
                        'x': pos[0],
                        'y': pos[1],
                        'speed': speed,
                        'angle': angle
                    })
        except Exception:
            pass
        return obstacles

    def _check_lateral_gap_rss(self, direction):
        """
        Checks if the lateral gap in the target lane is safe according to RSS.
        direction: 1 (left) or -1 (right)
        """
        try:
            ego_lane_idx = self.veh_data["lane_idx"]
            target_lane_idx = ego_lane_idx + direction
            ego_road_id = self.veh_data["road_id"]
            s_ego = self.veh_data["lane_pos"]
            v_ego = self.veh_data["speed"]
            
            # Ensure context subscription is active
            context = traci.vehicle.getContextSubscriptionResults(self.VEH_ID)
            if not context:
                traci.vehicle.subscribeContext(
                    self.VEH_ID,
                    tc.CMD_GET_VEHICLE_VARIABLE,
                    self.JUNCTION_CONTEXT_RANGE,
                    [tc.VAR_SPEED, tc.VAR_POSITION, tc.VAR_ANGLE]
                )
                context = traci.vehicle.getContextSubscriptionResults(self.VEH_ID)
                
            if not context:
                return True # No vehicles around, safe
                
            delta = self.RSS_DELTA
            a_max = self.MAX_ACCEL
            d_ego = self.MAX_DECEL
            d_leader = self.MAX_DECEL
            d_min = self.RSS_D_MIN
            
            for other_id in context.keys():
                if other_id == self.VEH_ID:
                    continue
                    
                try:
                    other_road_id = traci.vehicle.getRoadID(other_id)
                    other_lane_idx = traci.vehicle.getLaneIndex(other_id)
                except Exception:
                    continue
                    
                if other_road_id == ego_road_id and other_lane_idx == target_lane_idx:
                    try:
                        s_other = traci.vehicle.getLanePosition(other_id)
                        v_other = traci.vehicle.getSpeed(other_id)
                    except Exception:
                        continue
                        
                    if s_other > s_ego:
                        # Other vehicle is ahead of ego (other is leader, ego is follower)
                        d_actual = s_other - s_ego
                        v_ego_new = v_ego + a_max * delta
                        term_ego = (v_ego_new ** 2) / (2 * d_ego)
                        term_leader = (v_other ** 2) / (2 * d_leader)
                        d_safe = max(0.0, v_ego * delta + 0.5 * a_max * (delta ** 2) + term_ego - term_leader) + d_min
                        
                        if d_actual < d_safe:
                            return False
                    else:
                        # Other vehicle is behind ego (ego is leader, other is follower)
                        d_actual = s_ego - s_other
                        v_other_new = v_other + a_max * delta
                        term_follower = (v_other_new ** 2) / (2 * d_ego)
                        term_ego_lead = (v_ego ** 2) / (2 * d_leader)
                        d_safe = max(0.0, v_other * delta + 0.5 * a_max * (delta ** 2) + term_follower - term_ego_lead) + d_min
                        
                        if d_actual < d_safe:
                            return False
        except Exception:
            return True
            
        return True

    def _apply_action_planner(self, action):
        """
        Applies Kinematic-Guided Safe Action Planner.
        Modifies target_speed and lane change request based on RSS and lateral gap acceptance.
        Returns:
            final_action: np.array of shape (2,)
            is_overridden: bool
        """
        is_overridden = False
        raw_lane_change = action[0]
        raw_speed_control = action[1]
        
        target_speed = (raw_speed_control + 1.0) / 2.0 * self.MAX_SPEED
        final_lane_change = raw_lane_change
        
        if not self.veh_data:
            return action, False
            
        v_ego = self.veh_data["speed"]
        leader = self.veh_data["leader"]
        
        # --- A. Longitudinal Planning (RSS Longitudinal Safety Shield) ---
        if leader is not None:
            leader_id, d_actual = leader
            try:
                v_leader = traci.vehicle.getSpeed(leader_id)
            except Exception:
                v_leader = 0.0
                
            delta = self.RSS_DELTA
            a_max = self.MAX_ACCEL
            d_ego = self.MAX_DECEL
            d_leader = self.MAX_DECEL
            d_min = self.RSS_D_MIN
            
            v_ego_new = v_ego + a_max * delta
            term_ego = (v_ego_new ** 2) / (2 * d_ego)
            term_leader = (v_leader ** 2) / (2 * d_leader)
            
            d_safe = max(0.0, v_ego * delta + 0.5 * a_max * (delta ** 2) + term_ego - term_leader) + d_min
            
            if d_actual < d_safe:
                v_safe = max(0.0, v_leader - d_ego * (delta - (d_actual - d_min) / max(0.1, v_ego)))
                if target_speed > v_safe:
                    target_speed = v_safe
                    is_overridden = True
                    
        # --- B. Lateral Planning (Lateral Gap Acceptance Filter) ---
        if abs(raw_lane_change) > self.LANE_CHANGE_THRESHOLD:
            edge_id = self.veh_data["road_id"]
            if not edge_id.startswith(":"):
                current_lane = self.veh_data["lane_idx"]
                num_lanes = traci.edge.getLaneNumber(edge_id)
                direction = 1 if raw_lane_change > 0.0 else -1
                
                can_change_sumo = False
                if direction == 1 and current_lane < num_lanes - 1 and self.veh_data["can_left"] > 0.0:
                    can_change_sumo = True
                elif direction == -1 and current_lane > 0 and self.veh_data["can_right"] > 0.0:
                    can_change_sumo = True
                    
                is_lateral_safe = False
                if can_change_sumo:
                    is_lateral_safe = self._check_lateral_gap_rss(direction)
                    
                if not (can_change_sumo and is_lateral_safe):
                    final_lane_change = 0.0
                    is_overridden = True
            else:
                final_lane_change = 0.0
                is_overridden = True
                
        final_speed_control = (target_speed / self.MAX_SPEED) * 2.0 - 1.0
        final_speed_control = np.clip(final_speed_control, -1.0, 1.0)
        
        final_action = np.array([final_lane_change, final_speed_control], dtype=np.float32)
        return final_action, is_overridden

    def step(self, action):
        self.step_count += 1
        
        SIM_STEPS = 1

        if not self._veh_exists():
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, \
                {"real_speed": 0, "reason": "already_dead", "is_success": 0}

        # Initialize helper variables for tracking if not present
        if not hasattr(self, "override_count"):
            self.override_count = 0
        if not hasattr(self, "episode_jerk_sum"):
            self.episode_jerk_sum = 0.0
        if not hasattr(self, "_prev_accel"):
            self._prev_accel = 0.0

        # Apply action planner
        planned_action, is_overridden = self._apply_action_planner(action)
        if is_overridden:
            self.override_count += 1

        # 1. Decode actions
        # action[0] ∈ [-1, 1]: lane change intent (|val| > threshold → đổi làn)
        # action[1] ∈ [-1, 1]: speed control → [0, MAX_SPEED]
        target_speed = (planned_action[1] + 1.0) / 2.0 * self.MAX_SPEED

        # 2. Apply speed control — SUMO tự xử lý gia tốc/giảm tốc vật lý
        try:
            traci.vehicle.setSpeed(self.VEH_ID, max(0.0, target_speed))
        except Exception:
            pass

        # 3. Apply lane change — SUMO tự xử lý chuyển làn mượt mà
        try:
            edge_id = traci.vehicle.getRoadID(self.VEH_ID)
            if not edge_id.startswith(":"):  # Không đổi làn trong ngã tư
                current_lane = traci.vehicle.getLaneIndex(self.VEH_ID)
                num_lanes = traci.edge.getLaneNumber(edge_id)

                if planned_action[0] > self.LANE_CHANGE_THRESHOLD and current_lane < num_lanes - 1:
                    traci.vehicle.changeLane(self.VEH_ID, current_lane + 1, self.LANE_CHANGE_DURATION)
                elif planned_action[0] < -self.LANE_CHANGE_THRESHOLD and current_lane > 0:
                    traci.vehicle.changeLane(self.VEH_ID, current_lane - 1, self.LANE_CHANGE_DURATION)
                else:
                    # Giữ nguyên làn, hủy yêu cầu đổi làn trước đó (nếu có)
                    traci.vehicle.changeLane(self.VEH_ID, current_lane, self.LANE_CHANGE_DURATION)
        except Exception:
            pass

        reward             = 0.0
        terminated         = False
        truncated          = False
        accumulated_energy = 0.0
        sum_speed          = 0.0
        valid_steps        = 0
        termination_reason = "running"
        # Tích luỹ cost Lagrangian qua các SIM_STEPS (dùng cho info dict)
        _step_cs = 0.0
        _step_cc = 0.0
        _step_cr = 0.0

        for _ in range(SIM_STEPS):
            traci.simulationStep()
            self._update_cache()

            if self.veh_data is None:
                terminated = True
                teleport_list = traci.simulation.getStartingTeleportIDList()
                if self.VEH_ID in teleport_list:
                    reward -= 5.0
                    termination_reason = "teleport"
                else:
                    reward -= 20.0
                    termination_reason = "collision"
                break

            # Calculate physical jerk
            current_accel = self.veh_data["accel"]
            dt = traci.simulation.getDeltaT()
            if hasattr(self, "_prev_accel") and dt > 0:
                physical_jerk = abs(current_accel - self._prev_accel) / dt
            else:
                physical_jerk = 0.0
            self._prev_accel = current_accel
            self.episode_jerk_sum += physical_jerk

            # --- Thưởng thoát ngã tư an toàn (tốc độ < 2.0 m/s) ---
            current_in_junction = self.veh_data["road_id"].startswith(":")
            if getattr(self, '_prev_was_in_junction', False) and not current_in_junction:
                if self.veh_data["speed"] < 2.0:
                    reward += 0.1  # Thưởng nhỏ khi thoát ngã tư an toàn
            self._prev_was_in_junction = current_in_junction

            ego_speed = self.veh_data["speed"]
            leader    = self.veh_data["leader"]

            # Context Aware Stuck Detection
            is_red_light = False
            tls_data = self.veh_data["tls"]
            if tls_data and tls_data[0][2] < 20.0:
                state = tls_data[0][3].lower()
                if 'r' in state or 'y' in state: is_red_light = True

            # OPT 11: Dùng speed từ cache
            is_leader_stopped = (leader is not None and traci.vehicle.getSpeed(leader[0]) < 0.5)

            # Nhận diện rẽ trái nhường đường là lý do dừng chính đáng
            is_yielding_left_turn = self._is_left_turn_yielding()

            if ego_speed < 0.5 and not (is_red_light or is_leader_stopped or is_yielding_left_turn):
                reward -= 0.05
                self.stuck_time += 1
            else:
                self.stuck_time = 0

            sum_speed   += ego_speed
            valid_steps += 1
            e = max(0.0, self.veh_data["elec"])
            accumulated_energy += e if not np.isnan(e) else 0.0

            _rew_out = self._calculate_reward(planned_action)
            _r_task, _c_safety, _c_comfort, _c_redlight = self._split_reward_for_step(_rew_out)
            reward += _r_task / SIM_STEPS
            # Cộng dồn cost qua các SIM_STEPS (SIM_STEPS=1 hiện tại → tương đương gán)
            _step_cs += _c_safety
            _step_cc += _c_comfort
            _step_cr += _c_redlight

            if self._success_check():
                terminated = True
                reward += 40.0
                self._success = True
                termination_reason = "goal"
                break

        if not terminated:
            if self.stuck_time > 100:
                terminated = True
                reward -= 40.0
                termination_reason = "stuck_too_long"
            elif self.step_count >= self.MAX_EPISODE_STEPS:
                truncated = True
                termination_reason = "timeout"
                _, _, final_offset = self._turn_info_cache
                if abs(final_offset) > 0.01:
                    reward -= 5.0

        # OPT 12: Chỉ build route_info khi cần
        route_info = ""
        if (terminated or truncated) and termination_reason in ("stuck_too_long", "timeout", "teleport"):
            if hasattr(self, "current_route_edges"):
                route_info = " -> ".join(self.current_route_edges)

        obs           = self._get_obs()
        avg_real_speed = sum_speed / max(1, valid_steps)

        safety_val = 0.0
        if self.veh_data and self.veh_data["leader"]:
            dist = self.veh_data["leader"][1]
            if dist < 20: safety_val = 1.0 - (dist / 20.0)

        avg_jerk = self.episode_jerk_sum / max(1, self.step_count)
        override_rate = self.override_count / max(1, self.step_count)

        info = {
            "real_speed":  avg_real_speed,
            "real_energy": accumulated_energy,
            "safety":      safety_val,
            "comfort":     getattr(self, "last_jerk", 0.0),
            "wiggle":      getattr(self, "last_jerk", 0.0),
            "step_reward": reward,
            "is_success":  1 if self._success else 0,
            "reason":      termination_reason,
            "route":       route_info,
            "override_rate": override_rate,
            "avg_jerk":    avg_jerk,
            # Cost Lagrangian (chỉ có ý nghĩa khi adaptive_reward_enabled=True;
            # khi tắt luôn = 0 do _split_reward_for_step trả 0)
            "cost_safety":   _step_cs,
            "cost_comfort":  _step_cc,
            "cost_redlight": _step_cr,
        }

        return obs, reward, terminated, truncated, info

    def _success_check(self):
        if not self.veh_data: return False
        try:
            current_edge = self.veh_data["road_id"]
            if current_edge.startswith(":"): return False

            if hasattr(self, "current_route_edges") and self.current_route_edges:
                if current_edge == self.current_route_edges[-1]:
                    lane_len = traci.lane.getLength(self.veh_data["lane_id"])
                    pos      = self.veh_data["lane_pos"]
                    if pos > (lane_len - 20.0):
                        return True
        except: pass
        return False

    def close(self):
        try:
            traci.close()
        except:
            pass

if __name__ == "__main__":
    env = SumoEnv(map_config="TestMap/osm.sumocfg", render=True, test_mode=False, test_route="TestMap/test_route.rou.xml", delay=100)
    obs, info = env.reset()
    print(f"Init Obs Shape: {obs.shape}")
    total_reward = 0
    print("Starting Loop...")
    for i in range(50):
        action = env.action_space.sample()
        action[1] = np.random.uniform(-1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {env.step_count} | Action: {action} | Reward: {reward:.2f} | Speed: {obs[0]:.2f} | Energy: {obs[2]:.2f}")
        if terminated or truncated:
            print("Episode Finished!")
            obs, info = env.reset()
    env.close()
    print("Test Complete.")