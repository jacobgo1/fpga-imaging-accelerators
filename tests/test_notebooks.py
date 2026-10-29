"""The two board notebooks every model shares, end to end, for every model in tools/models.py.

software/pynq/shared/classify_golden.ipynb runs the golden kernels, classify.ipynb the
optimized ones. Their code cells run against a fake `pynq` whose kernel computes with the
PC's models (tools/models.py: golden_forward, fixed_scores), on an image folder made by
tools/image.py prepare, with the kernel.json main.py packs. The notebooks check the board
against the labels only; here their answers are checked against the PC's (golden_scores,
fixed_scores). That checks the register traffic, the preprocessing and patching, the strips
and the stitching -- not PYNQ or the board.
"""
import contextlib
import gc
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
import zipfile

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / 'software/pynq/shared'
GOLDEN_REGS = {'AP_CTRL': 0x00, 'DIN_DATA': 0x10, 'DOUT_DATA': 0x1c, 'N_DATA': 0x28}
OPT_REGS = {'AP_CTRL': 0x00, 'DIN0_DATA': 0x10, 'DIN1_DATA': 0x1c, 'DIN2_DATA': 0x28, 'DIN3_DATA': 0x34,
            'DOUT_DATA': 0x40, 'HEIGHT_DATA': 0x4c, 'WIDTH_DATA': 0x54, 'CYCLES_DATA': 0x5c}
CYCLES_PER_PIXEL = 7    # what the fake optimized kernel's counter says
MHZ = 200.0


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def golden_has_n(model):
    """Whether the golden kernel takes n units per start (an n argument), from its C++."""
    return ', int n)' in (ROOT / 'src/hls' / model / f'{model}.hpp').read_text(encoding='utf-8')


class FakeBuffer(np.ndarray if np is not None else object):
    """pynq.allocate's buffer: an array with a physical address."""

    def flush(self): pass
    def invalidate(self): pass


class FakeKernel:
    """The HLS block's AXI-Lite registers (regs: name -> offset). On start the golden kernel
    computes its n units with golden_forward, the optimized one the whole cube with
    fixed_scores. Allocations larger than max_bytes fail, as when PYNQ's CMA is short."""

    def __init__(self, models, model, golden, regs, max_bytes=None):
        self.models, self.model, self.golden, self.regs, self.max_bytes = models, model, golden, regs, max_bytes
        self.forward = models.golden_forward(model) if golden else None
        self.patch = models.MODELS[model]['patch']
        self.words, self.buffers, self.done, self.runs = {}, {}, False, []
        self.next_address = 0x7000_0000

    def allocate(self, shape, dtype):
        if self.max_bytes is not None and np.prod(shape) * np.dtype(dtype).itemsize > self.max_bytes:
            raise RuntimeError('Failed to allocate Memory!')
        buffer = np.zeros(shape, dtype=dtype).view(FakeBuffer)
        buffer.physical_address = self.next_address
        self.buffers[self.next_address] = buffer
        self.next_address += 0x1_0000_0000   # the second buffer lands above 4 GiB: the high word matters
        return buffer

    def address(self, name):
        offset = self.regs[name]
        return self.words.get(offset, 0) | (self.words.get(offset + 4, 0) << 32)

    def view(self, address, shape, dtype):
        """shape x dtype at a physical address, inside one buffer."""
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        for start, buffer in self.buffers.items():
            if start <= address and address + size <= start + buffer.nbytes:
                raw = np.asarray(buffer).reshape(-1).view(np.uint8)
                return raw[address - start:address - start + size].view(dtype).reshape(shape)
        raise AssertionError(f'0x{address:x}: no buffer holds {shape} {np.dtype(dtype)} there')

    def read(self, offset):
        if offset == self.regs['AP_CTRL']:
            done, self.done = self.done, False
            return (0x2 if done else 0) | 0x4
        return self.words.get(offset, 0)

    def write(self, offset, value):
        if offset != self.regs['AP_CTRL'] or not value & 1:
            self.words[offset] = value
            return
        if self.golden:
            n = self.words[self.regs['N_DATA']] if 'N_DATA' in self.regs else 1
            unit_in = (self.patch, self.patch, 110) if self.patch else (110,)
            unit_out = (self.patch, self.patch, 3) if self.patch else (3,)
            units = self.view(self.address('DIN_DATA'), (n,) + unit_in, np.float32)
            scores = (np.stack([self.forward(u) for u in units]) if self.patch else self.forward(units))
            self.view(self.address('DOUT_DATA'), (n,) + unit_out, np.float32)[:] = scores
            self.runs.append(n)
        else:
            dins = [name for name in self.regs if name.startswith('DIN')]
            assert len({self.address(d) for d in dins}) == 1, 'all input ports read the same cube'
            h, w = self.words[self.regs['HEIGHT_DATA']], self.words[self.regs['WIDTH_DATA']]
            out = self.models.fixed_scores(self.model, self.view(self.address('DIN0_DATA'), (h, w, 120), np.uint16))
            self.view(self.address('DOUT_DATA'), (h, w, 4), np.int32)[:] = out
            if 'CYCLES_DATA' in self.regs:
                self.words[self.regs['CYCLES_DATA']] = CYCLES_PER_PIXEL * h * w
            self.runs.append((h, w))
        self.done = True


@unittest.skipIf(np is None, 'needs numpy')
class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'tools'))
        cls.models = load_module('models', ROOT / 'tools/models.py')
        cls.preprocessing = load_module('preprocessing', ROOT / 'tools/preprocessing.py')
        cls.image = load_module('image', ROOT / 'tools/image.py')

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(ROOT / 'tools'))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        gc.collect()        # a notebook's namespace (a cycle) may still map cube.npy; Windows locks it
        self.tmp.cleanup()

    def prepare(self, model, kernel, regs, h, w, labels=True):
        """A board folder as main.py packs it for KERNEL (register map, kernel.json, mu_sd.txt)
        and an image zip next to it: a raw cube of counts whose z-scores are N(0, 1), the
        dropped bands 65535, so a wrong band selection shows. self.cube: that cube."""
        kept, mean, inv_std = self.preprocessing.read_preprocessing()
        z = np.random.default_rng(1).normal(0, 1, (h, w, 110))
        raw = np.full((h, w, 120), 65535.0)
        raw[:, :, kept] = z / inv_std + mean
        case = Path(tempfile.mkdtemp(dir=self.dir))      # a folder of its own: tests prepare several
        self.cube = np.clip(np.rint(raw), 0, 65535).astype(np.uint16)
        np.save(case / 'source.npy', self.cube)
        np.save(case / 'source_labels.npy', (np.arange(h * w).reshape(h, w) % 4).astype(np.uint8))
        board = case / 'board'
        board.mkdir()
        image = board / 'aegean_unet'
        saved, sys.argv = sys.argv, ['image.py', 'prepare', str(case / 'source.npy'), '--out', str(image),
                                     '--labels', str(case / 'source_labels.npy')]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(self.image.main(), 0)
        finally:
            sys.argv = saved
        with zipfile.ZipFile(board / 'aegean_unet.zip', 'w') as z:
            for f in image.iterdir():
                if labels or f.name != 'labels.npy':     # without: an image the notebooks refuse
                    z.write(f, f'aegean_unet/{f.name}')
        shutil.rmtree(image)
        (board / f'x{kernel}_hw.h').write_text(''.join(
            f'#define X{kernel.upper()}_CONTROL_ADDR_{reg} 0x{offset:02x}\n' for reg, offset in regs.items()))
        (board / 'kernel.json').write_text(json.dumps({'kernel': kernel, 'model': model, **self.models.MODELS[model]}))
        shutil.copy2(ROOT / 'mu_sd.txt', board)
        return board

    def run_notebook(self, notebook, board, kernel):
        """Every code cell of the notebook, in order, with fake pynq and matplotlib."""
        overlay = types.SimpleNamespace(kernel=kernel)
        pynq = types.ModuleType('pynq')
        pynq.Overlay = lambda path: overlay
        pynq.allocate = kernel.allocate
        pynq.Clocks = types.SimpleNamespace(fclk0_mhz=MHZ)
        pyplot = types.ModuleType('matplotlib.pyplot')

        class Anything:   # a figure or axes: every call does nothing
            def __getattr__(self, name):
                return lambda *a, **k: None
        pyplot.subplots = lambda n=1, m=1, **kw: (Anything(), [Anything() for _ in range(n * m)])
        pyplot.show = lambda *a, **k: None
        matplotlib = types.ModuleType('matplotlib')
        matplotlib.pyplot = pyplot
        fakes = {'pynq': pynq, 'matplotlib': matplotlib, 'matplotlib.pyplot': pyplot}
        saved = {name: sys.modules.get(name) for name in fakes}
        sys.modules.update(fakes)
        cwd, namespace, printed = Path.cwd(), {}, io.StringIO()
        try:
            os.chdir(board)
            with contextlib.redirect_stdout(printed):
                for cell in json.loads(notebook.read_text(encoding='utf-8'))['cells']:
                    if cell['cell_type'] == 'code':
                        exec(''.join(cell['source']), namespace)
        finally:
            os.chdir(cwd)
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        return namespace, printed.getvalue()

    def golden(self, model, h=40, w=50, max_bytes=None, labels=True):
        regs = dict(GOLDEN_REGS) if golden_has_n(model) else {k: v for k, v in GOLDEN_REGS.items() if k != 'N_DATA'}
        board = self.prepare(model, model, regs, h, w, labels)
        kernel = FakeKernel(self.models, model, True, regs, max_bytes)
        return (*self.run_notebook(SHARED / 'classify_golden.ipynb', board, kernel), kernel, board)

    def optimized(self, model, h=40, w=50, ports=4, counter=True, max_bytes=None, labels=True):
        regs = {k: v for k, v in OPT_REGS.items()
                if not (k.startswith('DIN') and int(k[3]) >= ports) and (counter or k != 'CYCLES_DATA')}
        board = self.prepare(model, f'{model}_opt', regs, h, w, labels)
        kernel = FakeKernel(self.models, model, False, regs, max_bytes)
        return (*self.run_notebook(SHARED / 'classify.ipynb', board, kernel), kernel, board)

    def assert_golden_answer(self, ns, model):
        """The notebook put each unit's scores in place: the PC's golden_scores for the cube."""
        want = self.models.golden_scores(model, self.cube)
        np.testing.assert_allclose(ns['scores'], want, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(ns['classes'], ns['scores'].argmax(-1))

    def assert_fixed_answer(self, ns, model):
        """What the optimized kernel wrote, read back whole: the PC's fixed_scores for the cube."""
        np.testing.assert_array_equal(ns['raw_scores'], self.models.fixed_scores(model, self.cube))
        np.testing.assert_array_equal(ns['classes'], ns['raw_scores'][..., 3])

    def assert_checked_against_the_labels(self, ns, printed):
        self.assertIn('accuracy against the labels', printed)
        self.assertIn('differs from labels', ns['panels'])
        known = ns['labels'] < 3
        self.assertIn(f'{(ns["classes"][known] == ns["labels"][known]).mean():.2%}', printed)

    # ---- the golden notebook ----

    def test_golden_notebook_classifies_every_model_like_the_pc(self):
        for model in self.models.MODELS:
            with self.subTest(model=model):
                ns, printed, kernel, board = self.golden(model)
                self.assertEqual(ns['scores'].shape, (40, 50, 3))
                self.assert_golden_answer(ns, model)
                self.assertLess(float(np.abs(ns['scores']).max()), 1000, 'dropped bands reached the kernel')
                patch = self.models.MODELS[model]['patch']
                if not patch:
                    self.assertEqual(kernel.runs, [256, 40 * 50], 'the timed first pixels, then all of them')
                elif golden_has_n(model):
                    self.assertEqual(kernel.runs, [1, 4], 'the timed first patch, then 2 x 2 patches in one start')
                else:
                    self.assertEqual(kernel.runs, [1] * 5, 'the timed first patch, then one start per patch')
                self.assertTrue((board / f'aegean_unet_{model}_fpga_scores.npy').is_file())
                self.assert_checked_against_the_labels(ns, printed)

    def test_golden_notebook_goes_band_by_band_when_cma_is_short(self):
        # sp_unet_small: 3 x 2 patches of 32 x 32 x 110 floats (2.7 MB) do not fit, a row of 2 does.
        ns, _, kernel, _ = self.golden('sp_unet_small', h=70, w=50, max_bytes=1_000_000)
        self.assertEqual(ns['BANDS_PER_START'], 1)
        self.assertEqual(kernel.runs, [1, 2, 2, 2])
        self.assert_golden_answer(ns, 'sp_unet_small')
        # justoliunet: 128-row bands of pixels
        ns, _, kernel, _ = self.golden('justoliunet', h=150, w=40, max_bytes=2_300_000)
        self.assertEqual(kernel.runs[1:], [128 * 40, 22 * 40])
        self.assert_golden_answer(ns, 'justoliunet')

    def test_golden_notebook_needs_the_labels(self):
        with self.assertRaisesRegex(FileNotFoundError, 'no labels.npy'):
            self.golden('spectralpointwise1d', labels=False)

    # ---- the optimized notebook ----

    def test_optimized_notebook_classifies_every_model_bit_for_bit(self):
        for model in self.models.MODELS:
            with self.subTest(model=model):
                ns, printed, kernel, board = self.optimized(model)
                self.assertEqual(ns['scores'].shape, (40, 50, 3))
                self.assert_fixed_answer(ns, model)
                self.assertEqual(kernel.runs, [(40, 50)] * 6, 'the whole image, then 5 timed runs of it')
                self.assertEqual(ns['DINS'], [OPT_REGS[f'DIN{i}_DATA'] for i in range(4)])
                for din in ns['DINS']:
                    self.assertEqual(kernel.words[din], ns['cube_in'].physical_address & 0xFFFFFFFF)
                self.assertAlmostEqual(ns['own'], CYCLES_PER_PIXEL * 40 * 50 / (MHZ * 1e6))
                self.assertEqual(ns['UNIT'], 'patch' if self.models.MODELS[model]['patch'] else 'pixel')
                self.assertTrue((board / f'aegean_unet_{model}_opt_fpga_scores.npy').is_file())
                self.assert_checked_against_the_labels(ns, printed)

    def test_optimized_notebook_runs_a_board_version_without_the_counter(self):
        """The Zynq-7000 versions: fewer read ports, no cycles register."""
        ns, _, _, _ = self.optimized('waveletcnn', ports=2, counter=False)
        self.assertEqual(len(ns['DINS']), 2)
        self.assertIsNone(ns['CYCLES'])
        self.assertIsNone(ns['own'])
        self.assert_fixed_answer(ns, 'waveletcnn')

    def test_optimized_notebook_uses_strips_when_the_cube_does_not_fit(self):
        # 150 x 40 x 120 x 2 bytes = 1.44 MB does not fit, a 128-row strip (1.23 MB) does.
        ns, _, kernel, _ = self.optimized('justounetsimple', h=150, w=40, max_bytes=1_300_000)
        self.assertIsNone(ns['cube_in'])
        self.assertEqual(kernel.runs[:2], [(128, 40), (22, 40)], 'two strips, the second the rest')
        self.assert_fixed_answer(ns, 'justounetsimple')
        self.assertAlmostEqual(ns['own'], CYCLES_PER_PIXEL * 150 * 40 / (MHZ * 1e6), msg='added up over the strips')

    def test_optimized_notebook_finds_the_shared_image_in_the_folder_above(self):
        board = self.prepare('justoliunet_bn', 'justoliunet_bn_opt', OPT_REGS, 40, 40)
        shutil.move(str(board / 'aegean_unet.zip'), str(board.parent / 'aegean_unet.zip'))
        kernel = FakeKernel(self.models, 'justoliunet_bn', False, OPT_REGS)
        ns, _ = self.run_notebook(SHARED / 'classify.ipynb', board, kernel)
        self.assert_fixed_answer(ns, 'justoliunet_bn')
        self.assertTrue((board.parent / 'aegean_unet/cube.npy').is_file(), 'unpacked next to its zip')
        self.assertFalse((board / 'aegean_unet').exists(), "no copy in the design's folder")

    def test_optimized_notebook_needs_the_labels_before_the_fpga_runs(self):
        board = self.prepare('justounetsimple', 'justounetsimple_opt', OPT_REGS, 40, 50, labels=False)
        kernel = FakeKernel(self.models, 'justounetsimple', False, OPT_REGS)
        with self.assertRaisesRegex(FileNotFoundError, 'no labels.npy'):
            self.run_notebook(SHARED / 'classify.ipynb', board, kernel)
        self.assertEqual(kernel.runs, [], 'stopped before the first start')

    # ---- the table ----

    def test_every_kernel_belongs_to_a_model_or_has_its_own_notebook(self):
        for source in sorted((ROOT / 'src/hls').iterdir()):
            with self.subTest(kernel=source.name):
                model = self.models.model_of(source.name)
                if model is None:
                    self.assertTrue((ROOT / 'software/pynq' / source.name).is_dir(), 'no notebook for it')
        self.assertEqual(self.models.model_of('sp_unet_small_opt_direct_zynq7020'), 'sp_unet_small')
        self.assertEqual(self.models.model_of('justoliunet_bn_opt'), 'justoliunet_bn')
        self.assertEqual(self.models.model_of('justoliunet_opt'), 'justoliunet')

    def test_the_table_matches_the_models(self):
        for model, entry in self.models.MODELS.items():
            with self.subTest(model=model):
                self.assertTrue((ROOT / 'src/hls' / model).is_dir(), 'its golden kernel')
                if model.startswith('sp_unet_'):
                    module = load_module('specpw_unet_model', ROOT / 'tools/specpw_unet_model.py')
                elif entry['patch'] is None:
                    name = 'spectralpointwise1d_model' if model == 'spectralpointwise1d' else 'justoliunet_model'
                    module = load_module(name, ROOT / 'tools' / f'{name}.py')
                else:
                    module = load_module(f'{model}_model', ROOT / 'tools' / f'{model}_model.py')
                self.assertEqual(entry['out_frac'], module.OUT_FRAC)


if __name__ == '__main__':
    unittest.main()
