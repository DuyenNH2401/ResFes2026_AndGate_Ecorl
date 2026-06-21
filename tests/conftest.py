"""Test fixtures cho môi trường KHÔNG có SUMO/TraCI.

Inject một module `traci` giả vào sys.modules trước khi import simulation.env,
để test logic reward chạy được mà không cần cài SUMO. KHÔNG ảnh hưởng code
production (chỉ tác động trong quá trình chạy pytest).
"""
import os
import sys
import types


def _install_fake_traci():
    if "traci" in sys.modules:
        return
    traci = types.ModuleType("traci")
    traci.__path__ = []  # đánh dấu là package để cho phép `import traci.constants`

    class _Stub:
        def __getattr__(self, name):
            def _f(*args, **kwargs):
                raise RuntimeError("fake traci called: %s" % name)
            return _f

    # các submodule traci.* mà env.py truy cập
    for sub in ("vehicle", "lane", "trafficlight", "simulation", "route", "edge"):
        setattr(traci, sub, _Stub())
    traci.close = lambda *a, **k: None
    traci.start = lambda *a, **k: None

    # traci.constants — module rỗng có __file__ hợp lệ để torch/inspect không lỗi.
    constants = types.ModuleType("traci.constants")
    constants.__file__ = os.path.join(os.path.dirname(__file__), "_fake_traci_constants.py")
    constants.__getattr__ = lambda name: 0  # bất kỳ hằng nào → 0
    traci.constants = constants
    traci.__file__ = os.path.join(os.path.dirname(__file__), "_fake_traci.py")

    sys.modules["traci"] = traci
    sys.modules["traci.constants"] = constants


_install_fake_traci()

# env.py thoát nếu thiếu SUMO_HOME; đặt giá trị giả cho môi trường test.
os.environ.setdefault("SUMO_HOME", "/tmp/_fake_sumo_home")
