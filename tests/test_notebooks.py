"""The two board notebooks every model shares, end to end, for every model in tools/models.py.

software/pynq/classify_golden.ipynb runs the golden kernels, classify.ipynb the
optimized ones. Their code cells run against a fake `pynq` whose kernel computes with the
PC's models (tools/models.py: golden_forward, fixed_scores), on an image folder made by
tools/image.py prepare, with the kernel.json scripts/pynq.sh packs. The notebooks check the board
against the labels only; here their answers are checked against the PC's (golden_scores,
fixed_scores). That checks the register traffic, the preprocessing and patching, the strips
and the stitching -- not PYNQ or the board.
"""
import contextlib
import gc
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

from helpers import ROOT, kernel_dir, kernel_dirs, load_module


try:
    import numpy as np
except ImportError:
    np = None

SHARED = ROOT / 'software/pynq'
GOLDEN_REGS = {'AP_CTRL': 0x00, 'DIN_DATA': 0x10, 'DOUT_DATA': 0x1c, 'N_DATA': 0x28}
OPT_REGS = {'AP_CTRL': 0x00, 'DIN0_DATA': 0x10, 'DIN1_DATA': 0x1c, 'DIN2_DATA': 0x28, 'DIN3_DATA': 0x34,
            'DOUT_DATA': 0x40, 'HEIGHT_DATA': 0x4c, 'WIDTH_DATA': 0x54, 'CYCLES_DATA': 0x5c}
CYCLES_PER_PIXEL = 7    # what the fake optimized kernel's counter says
MHZ = 200.0


def golden_has_n(model):
    """Whether the golden kernel takes n units per start (an n argument), from its C++."""
    return ', int n)' in (kernel_dir(model) / f'{model}.hpp').read_text(encoding='utf-8')


class FakeBuffer(np.ndarray if np is not None else object):
    """pynq.allocate's buffer: an array with a physical address."""

    def flush(self): pass
    def invalidate(self): pass
    def freebuffer(self): pass


class FakeKernel:
    """The HLS block's AXI-Lite registers (regs: name -> offset). On start the golden kernel
    computes its n units with golden_forward, the optimized one the whole cube with
    fixed_scores. Allocations larger than max_bytes fail, as when PYNQ's CMA is short."""

    def __init__(self, models, model, golden, regs, max_bytes=None, kernel=None):
        self.models, self.model, self.golden, self.regs, self.max_bytes = models, model, golden, regs, max_bytes
        self.kernel = kernel   # the optimized kernel's name, for fixed_scores
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
            out = self.models.fixed_scores(self.model, self.view(self.address('DIN0_DATA'), (h, w, 120), np.uint16),
                                           self.kernel)
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
        """A board folder as scripts/pynq.sh packs it for KERNEL (register map, kernel.json, mu_sd.txt)
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
        # The same cube as a capture stores it, for the OPU boards' cnn_accelerator.
        self.assertEqual((image / 'binned_cube.bip').read_bytes(), self.cube.tobytes())
        self.assertEqual((image / 'capture_config.ini').read_text(),
                         f'frame_count = {h}\nrow_count = {w}\naoi_y = 0\nexposure = 0.0\n')
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

    def run_notebook(self, notebook, board, kernel, overlays=None):
        """Every code cell of the notebook, in order, with fake pynq and matplotlib. overlays:
        a bitstream's path -> its kernel, for a notebook that loads several; kernel allocates."""
        overlay = types.SimpleNamespace(kernel=kernel)
        pynq = types.ModuleType('pynq')
        pynq.Overlay = ((lambda path: types.SimpleNamespace(kernel=overlays(path))) if overlays
                        else (lambda path: overlay))
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

    def optimized(self, model, h=40, w=50, ports=4, counter=True, max_bytes=None, labels=True, name=None):
        regs = {k: v for k, v in OPT_REGS.items()
                if not (k.startswith('DIN') and int(k[3]) >= ports) and (counter or k != 'CYCLES_DATA')}
        name = name or f'{model}_opt'
        board = self.prepare(model, name, regs, h, w, labels)
        kernel = FakeKernel(self.models, model, False, regs, max_bytes, name)
        return (*self.run_notebook(SHARED / 'classify.ipynb', board, kernel), kernel, board)

    def assert_golden_answer(self, ns, model):
        """The notebook put each unit's scores in place: the PC's golden_scores for the cube."""
        want = self.models.golden_scores(model, self.cube)
        np.testing.assert_allclose(ns['scores'], want, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(ns['classes'], ns['scores'].argmax(-1))

    def assert_fixed_answer(self, ns, model, kernel=None):
        """What the optimized kernel wrote, read back whole: the PC's fixed_scores for the cube."""
        np.testing.assert_array_equal(ns['raw_scores'], self.models.fixed_scores(model, self.cube, kernel))
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

    def test_optimized_notebook_classifies_a_uint8_kernel_bit_for_bit(self):
        """sp_unet_small_opt_u8 shares the notebook; it answers with uint8 activations, not
        what the int16 kernels answer."""
        ns, printed, _, board = self.optimized('sp_unet_small', name='sp_unet_small_opt_u8')
        self.assert_fixed_answer(ns, 'sp_unet_small', 'sp_unet_small_opt_u8')
        self.assertFalse((ns['raw_scores'] == self.models.fixed_scores('sp_unet_small', self.cube)).all(),
                         'the uint8 arithmetic is not the int16 one')
        self.assertTrue((board / 'aegean_unet_sp_unet_small_opt_u8_fpga_scores.npy').is_file())
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

    # ---- the benchmark notebook ----

    def benchmark_board(self, models, h=20, w=24):
        """The Jupyter folder: KERNEL_pynq/ for MODEL_opt of each of models, and the benchmark's
        five images as zips. Returns (the folder, the fake kernels by name with the allocator
        under None, each image's cube and labels)."""
        top = self.dir / 'jupyter'
        top.mkdir()
        allocator = FakeKernel(self.models, models[0], False, OPT_REGS)
        kernels = {None: allocator}
        for model in models:
            name = f'{model}_opt'
            folder = top / f'{name}_pynq'
            folder.mkdir()
            (folder / f'{name}.bit').write_bytes(b'')
            (folder / f'x{name}_hw.h').write_text(''.join(
                f'#define X{name.upper()}_CONTROL_ADDR_{reg} 0x{offset:02x}\n' for reg, offset in OPT_REGS.items()))
            (folder / 'kernel.json').write_text(json.dumps({'kernel': name, 'model': model, **self.models.MODELS[model]}))
            kernels[name] = FakeKernel(self.models, model, False, OPT_REGS, kernel=name)
            kernels[name].buffers = allocator.buffers     # one DDR: every kernel sees every buffer
        images = {}
        rng = np.random.default_rng(2)
        for name in ('trondheim', 'hawkebay', 'lakealexandria', 'saldanha', 'maputo'):
            cube = rng.integers(0, 4000, (h, w, 120)).astype(np.uint16)
            labels = rng.integers(0, 4, (h, w)).astype(np.uint8)
            with zipfile.ZipFile(top / f'{name}.zip', 'w') as z:
                for file, array in (('cube.npy', cube), ('labels.npy', labels)):
                    data = io.BytesIO()
                    np.save(data, array)
                    z.writestr(f'{name}/{file}', data.getvalue())
                z.writestr(f'{name}/meta.json', json.dumps({'rgb_bands': [56, 67, 86]}))
            images[name] = (cube, labels)
        return top, kernels, images

    def benchmark(self, top, kernels):
        return self.run_notebook(SHARED / 'benchmark.ipynb', top, kernels[None],
                                 overlays=lambda path: kernels[Path(path).stem])

    def test_benchmark_notebook_runs_each_kernel_on_each_image_and_writes_the_report(self):
        top, kernels, images = self.benchmark_board(['justounetsimple', 'spectralpointwise1d'])
        ns, _ = self.benchmark(top, kernels)
        self.assertEqual(ns['found'], ['justounetsimple_opt', 'spectralpointwise1d_opt'])
        for name in ns['found']:
            self.assertEqual(kernels[name].runs, [(20, 24)] * 5, 'each image once, whole')
        self.assertEqual(len(ns['rows']), 10)
        for row in ns['rows']:
            with self.subTest(image=row['image'], kernel=row['kernel']):
                self.assertIsNone(row['error'])
                cube, labels = images[row['image']]
                classes = self.models.fixed_scores(self.models.model_of(row['kernel']), cube, row['kernel'])[..., 3]
                known = labels < 3
                self.assertEqual(row['correct'], int((classes == labels)[known].sum()), "the kernel's own classes")
                self.assertEqual(row['labelled'], int(known.sum()))
                self.assertAlmostEqual(row['own_ms'], 1000 * CYCLES_PER_PIXEL * 20 * 24 / (MHZ * 1e6))
        # The report: a line per kernel with its means, every run, the kernels not uploaded.
        reports = sorted((top / 'benchmark_results').glob('benchmark_*.md'))
        self.assertEqual(len(reports), 1)
        report = reports[0].read_text()
        self.assertEqual(report, ns['report'])
        for s in ns['summary']:
            runs = [r for r in ns['rows'] if r['kernel'] == s['kernel']]
            pooled = sum(r['correct'] for r in runs) / sum(r['labelled'] for r in runs)
            self.assertIn(f'| `{s["kernel"]}` | {MHZ:.1f} | 5 |', report)
            self.assertIn(f'{pooled:.2%}', report)
        self.assertIn(f'{CYCLES_PER_PIXEL * 20 * 24:.2f} a patch', report)   # 20 x 24 pixels: one patch
        self.assertIn(f'{CYCLES_PER_PIXEL:.2f} a pixel', report)
        self.assertIn('Skipped, no `KERNEL_pynq/KERNEL.bit`: `justoliunet_opt`', report)
        with open(reports[0].with_suffix('.csv'), newline='') as f:
            self.assertEqual(len(f.read().strip().splitlines()), 1 + 10, 'a header and every run')

    def test_benchmark_notebook_goes_on_after_a_run_fails(self):
        top, kernels, _ = self.benchmark_board(['waveletcnn', 'justoliunet'])
        def broken(offset, value):
            raise TimeoutError('kernel did not finish')
        kernels['waveletcnn_opt'].write = broken
        ns, _ = self.benchmark(top, kernels)
        self.assertEqual([r['kernel'] for r in ns['rows'] if r['error']], ['waveletcnn_opt'] * 5)
        self.assertEqual(kernels['justoliunet_opt'].runs, [(20, 24)] * 5)
        self.assertEqual([s['kernel'] for s in ns['summary']], ['justoliunet_opt'])
        self.assertIn('failed: TimeoutError', ns['report'])

    def test_benchmark_notebook_lists_every_zcu104_kernel(self):
        cell = next(''.join(c['source']) for c in json.loads((SHARED / 'benchmark.ipynb').read_text(encoding='utf-8'))['cells']
                    if 'KERNELS = ' in ''.join(c['source']))
        namespace = {}
        exec(cell[cell.index('KERNELS = '):cell.index('# The images')], namespace)
        self.assertEqual(sorted(namespace['KERNELS']),
                         sorted(k for k in kernel_dirs() if k.endswith(('_opt', '_opt_u8', '_opt_tiled'))))

    # ---- the table ----

    def test_every_kernel_belongs_to_a_model_or_has_its_own_notebook(self):
        for name in sorted(kernel_dirs()):
            with self.subTest(kernel=name):
                model = self.models.model_of(name)
                if model is None:
                    self.assertTrue((ROOT / 'software/pynq' / name).is_dir(), 'no notebook for it')
        self.assertEqual(self.models.model_of('sp_unet_small_opt_zynq7020'), 'sp_unet_small')
        self.assertEqual(self.models.model_of('justoliunet_bn_opt'), 'justoliunet_bn')
        self.assertEqual(self.models.model_of('justoliunet_opt'), 'justoliunet')

    def test_the_table_matches_the_models(self):
        for model, entry in self.models.MODELS.items():
            with self.subTest(model=model):
                self.assertIn(model, kernel_dirs(), 'its golden kernel')
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
