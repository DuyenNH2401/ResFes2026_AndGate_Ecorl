#!/usr/bin/env python
"""
Kaggle environment setup for ecorl_adaptive_shaping.

Run this in the first cell of your Kaggle Notebook:
    !python setup_kaggle.py

Or copy-paste the relevant sections into a shell cell.
"""

import subprocess
import sys
import os

def run(cmd, hide_output=False, check=True):
    """Run a shell command."""
    if hide_output and check:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, shell=True, check=check)


def setup_sumo():
    """Install SUMO simulator via apt-get."""
    print("=" * 60)
    print("1/3  Installing SUMO simulator...")
    print("=" * 60)
    run("apt-get update -qq")
    run("apt-get install -y -qq sumo sumo-tools sumo-doc", hide_output=True)

    # Set SUMO_HOME environment variable
    sumo_home = "/usr/share/sumo"
    os.environ["SUMO_HOME"] = sumo_home

    # Persist for future cells
    run(f"echo 'SUMO_HOME={sumo_home}' >> /etc/environment")

    print(f"SUMO_HOME = {sumo_home}")
    print("SUMO installed successfully!\n")


def setup_python_deps():
    """Install Python dependencies."""
    print("=" * 60)
    print("2/3  Installing Python dependencies...")
    print("=" * 60)

    # No version pins — Kaggle's base env is recent enough.
    deps = [
        "torch",
        "numpy",
        "gymnasium",
        "traci",
        "pyyaml",
        "pandas",
        "matplotlib",
    ]

    # Check if torch is already installed (Kaggle usually has it)
    try:
        import torch  # noqa: F401
        print("  PyTorch already installed — skipping torch install")
        deps = [d for d in deps if not d.startswith("torch")]
    except ImportError:
        pass

    if deps:
        print(f"  Installing: {' '.join(deps)}")
        pip_cmd = f"{sys.executable} -m pip install {' '.join(deps)} -q"
        # Show stderr so we can debug if something goes wrong
        result = subprocess.run(
            pip_cmd, shell=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            print("  ⚠ pip install failed — trying one-by-one...")
            for dep in deps:
                retry_cmd = f"{sys.executable} -m pip install {dep} -q"
                ret = subprocess.run(retry_cmd, shell=True, capture_output=True, text=True)
                if ret.returncode == 0:
                    print(f"  ✓ {dep}")
                else:
                    print(f"  ✗ {dep}: {ret.stderr.strip()}")
        else:
            for dep in deps:
                print(f"  ✓ {dep}")
    else:
        print("  All dependencies already installed.")

    print("Python dependencies installed successfully!\n")


def verify():
    """Verify installation by running a quick import check."""
    print("=" * 60)
    print("3/3  Verifying installation...")
    print("=" * 60)

    packages = [
        ("PyTorch", "import torch; print(f'  ✓ torch {torch.__version__}')"),
        ("NumPy", "import numpy as np; print(f'  ✓ numpy {np.__version__}')"),
        ("Gymnasium", "import gymnasium as gym; print(f'  ✓ gymnasium {gym.__version__}')"),
        ("traci", "import traci; print(f'  ✓ traci {traci.__version__}')"),
        ("PyYAML", "import yaml; print(f'  ✓ yaml {yaml.__version__}')"),
        ("Pandas", "import pandas as pd; print(f'  ✓ pandas {pd.__version__}')"),
        ("Matplotlib", "import matplotlib; print(f'  ✓ matplotlib {matplotlib.__version__}')"),
        ("SUMO_HOME", "import os; v=os.getenv('SUMO_HOME','MISSING!'); print(f'  ✓ SUMO_HOME={v}')"),
    ]

    all_ok = True
    for name, check in packages:
        try:
            exec(check)
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            all_ok = False

    print()
    if all_ok:
        print("✓ All dependencies installed and verified!")
    else:
        print("⚠ Some dependencies failed — check messages above.")

    return all_ok


if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   ecorl_adaptive_shaping — Kaggle Environment Setup     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    setup_sumo()
    setup_python_deps()
    ok = verify()

    print()
    print("Setup complete! You can now run training.")
    print()
    print("  import os")
    print("  os.environ['SUMO_HOME'] = '/usr/share/sumo'")
    print("  !python train_ppg.py --backbone mamba [--lagrangian-enabled true ...]")
    print()

    sys.exit(0 if ok else 1)
