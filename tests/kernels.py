"""Where each kernel's C++ is, by main.py's lookup, for the tests that compile kernels."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location('main', Path(__file__).resolve().parents[1] / 'main.py')
_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_main)
kernel_dirs = _main.kernel_dirs


def kernel_dir(name):
    return kernel_dirs()[name]
