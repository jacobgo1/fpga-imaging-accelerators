"""What the tests share: the repository, the compilers, loading tools/, and compiling a kernel."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import unittest

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[1]
CXX = shutil.which('g++') or shutil.which('clang++')
CC = shutil.which('gcc') or shutil.which('clang')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kernel_dirs = load_module('main', ROOT / 'main.py').kernel_dirs


def kernel_dir(name):
    return kernel_dirs()[name]


def skip_without_aegean():
    if not (ROOT / 'aegean').is_dir():
        raise unittest.SkipTest('no aegean/ (not in git): copy the labelled test image in to run these')


def compile_cpp(tmp, stem, text, include_dirs=(), sources=()):
    """TEXT (C++ with a main()) and SOURCES, compiled with g++: the executable."""
    source = tmp / f'{stem}.cpp'
    source.write_text(text)
    exe = tmp / f'{stem}.exe'
    subprocess.run([CXX, '-std=c++14', '-O2', *[f'-I{d}' for d in (*include_dirs, ROOT / 'src/common')],
                    str(source), *[str(s) for s in sources], '-o', str(exe)], check=True)
    return exe


def run_kernel(tmp, kernel, driver, inputs, out_dtype, out_shape):
    """KERNEL with DRIVER, a main() around it, run as: exe INPUTS OUT and out_shape's sizes
    but the last; what it wrote to OUT."""
    source_dir = kernel_dir(kernel)
    exe = compile_cpp(tmp, f'run_{kernel}', driver, [source_dir], [source_dir / f'{kernel}.cpp'])
    out = tmp / f'out_{kernel}.bin'
    subprocess.run([str(exe), str(inputs), str(out), *[str(v) for v in out_shape[:-1]]], check=True)
    return np.fromfile(out, dtype=out_dtype).reshape(out_shape)


def raw_cube(model, h, w, seed=0, counts=False):
    """Raw values whose preprocessing gives N(0, 1); the dropped bands hold 1e6, so a
    wrong band selection blows the scores up. counts: whole numbers 0 .. 65535 like a
    capture's, the dropped bands 65535."""
    kept, mean, inv_std = model.read_preprocessing()
    z = np.random.default_rng(seed).normal(0, 1, (h, w, 110)).astype(np.float32)
    raw = np.full((h, w, 120), 65535.0 if counts else 1.0e6, dtype=np.float32)
    raw[:, :, kept] = z / inv_std + mean
    return np.clip(np.rint(raw), 0, 65535) if counts else raw
