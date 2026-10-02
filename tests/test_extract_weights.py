"""tools/extract_weights.py on the real checkpoints in weights/."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = sorted((ROOT / 'weights').rglob('*.pt'))
CXX = shutil.which('g++') or shutil.which('clang++')


@unittest.skipIf(np is None, 'the extractor needs numpy')
@unittest.skipUnless(CHECKPOINTS, 'no checkpoints in weights/')
class ExtractWeightsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'tools'))
        spec = importlib.util.spec_from_file_location('extract_weights', ROOT / 'tools/extract_weights.py')
        cls.tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.tool)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        result = subprocess.run([sys.executable, str(ROOT / 'tools/extract_weights.py'), '--out', str(cls.out)],
                                capture_output=True, text=True, cwd=ROOT)
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(ROOT / 'tools'))
        cls.tmp.cleanup()

    def models(self):
        return sorted(p for p in self.out.iterdir() if p.is_dir())

    def test_every_checkpoint_gets_header_arrays_and_summary(self):
        self.assertEqual(len(self.models()), len(CHECKPOINTS))
        for folder in self.models():
            for suffix in ('_weights.hpp', '_weights.npz', '.md'):
                self.assertTrue((folder / f'{folder.name}{suffix}').is_file(), f'{folder.name}{suffix}')
            self.assertIn(f'[{folder.name}]', (self.out / 'README.md').read_text())

    def test_arrays_hold_exactly_the_trained_values(self):
        import pt_reader
        for path in CHECKPOINTS:
            model, _, _ = self.tool.load(path)
            with self.subTest(model=model):
                state = pt_reader.state_dict(path)
                saved = np.load(self.out / model / f'{model}_weights.npz')
                expected = {k for k in state if not k.endswith('num_batches_tracked')}
                self.assertEqual(set(saved.files), expected)
                for name in expected:
                    np.testing.assert_array_equal(saved[name], np.asarray(state[name], dtype=np.float32))

    def test_extracting_one_model_keeps_the_others_in_the_overview(self):
        one = next(p for p in CHECKPOINTS if 'waveletcnn' in p.name)
        result = subprocess.run([sys.executable, str(ROOT / 'tools/extract_weights.py'), str(one),
                                 '--out', str(self.out)], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        overview = (self.out / 'README.md').read_text()
        for folder in self.models():
            self.assertIn(f'[{folder.name}]', overview)

    def test_layer_names_become_cpp_identifiers(self):
        self.assertEqual(self.tool.identifier('double1.block.0.weight'), 'double1_block_0_weight')
        self.assertEqual(self.tool.identifier('0.weight'), '_0_weight')

    @unittest.skipIf(CXX is None, 'needs g++ or clang++')
    def test_all_headers_compile_together_and_read_back_exactly(self):
        model, _, tensors = self.tool.load(next(p for p in CHECKPOINTS if 'justoliunet_pixel' in p.name))
        first = next(iter(tensors))
        includes = ''.join(f'#include "{m.name}/{m.name}_weights.hpp"\n' for m in self.models())
        source = (includes + '#include <cstdio>\n'
                  f'int main() {{ std::printf("%.9g", (double){model}::{self.tool.identifier(first)}'
                  + '[0]' * tensors[first].ndim + '); return 0; }\n')
        (self.out / 'check.cpp').write_text(source)
        exe = self.out / 'check.exe'
        build = subprocess.run([CXX, '-std=c++14', '-I', str(self.out), str(self.out / 'check.cpp'), '-o', str(exe)],
                               capture_output=True, text=True)
        self.assertEqual(build.returncode, 0, build.stderr[-2000:])
        printed = subprocess.run([str(exe)], capture_output=True, text=True).stdout
        self.assertEqual(np.float32(float(printed)), tensors[first].reshape(-1)[0])


if __name__ == '__main__':
    unittest.main()
