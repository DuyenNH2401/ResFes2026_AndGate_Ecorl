"""Tests cho adaptive KL (PPG paper): điều chỉnh β quanh d_targ."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppg.core import PPGAgent  # noqa: E402


def test_beta_increases_when_kl_too_high():
    # kl = 0.1 >> d_targ*thresh = 0.03*1.5 = 0.045 → β×2
    beta = PPGAgent._adapt_beta_kl(0.1, 5.0, 0.03, 1.5, 0.5, 100.0)
    assert beta == 10.0


def test_beta_decreases_when_kl_too_low():
    # kl = 0.001 << d_targ/thresh = 0.02 → β÷2
    beta = PPGAgent._adapt_beta_kl(0.001, 5.0, 0.03, 1.5, 0.5, 100.0)
    assert beta == 2.5


def test_beta_unchanged_in_deadband():
    # kl = 0.03 nằm trong [0.02, 0.045] → giữ nguyên
    beta = PPGAgent._adapt_beta_kl(0.03, 5.0, 0.03, 1.5, 0.5, 100.0)
    assert beta == 5.0


def test_beta_clamped_at_max():
    beta = PPGAgent._adapt_beta_kl(10.0, 80.0, 0.03, 1.5, 0.5, 100.0)
    assert beta == 100.0  # 80*2=160 → clamp 100


def test_beta_clamped_at_min():
    beta = PPGAgent._adapt_beta_kl(0.0, 0.6, 0.03, 1.5, 0.5, 100.0)
    assert beta == 0.5  # 0.6/2=0.3 → clamp 0.5
