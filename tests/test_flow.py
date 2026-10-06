"""main.py without AMD tools: finding a kernel's files, run folders, settings, the report."""
import contextlib
import importlib.util
import io
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

try:
    import tkinter
except ImportError:
    tkinter = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('flow', ROOT / 'main.py')
flow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flow)


def cli(*args):
    return subprocess.run([sys.executable, str(ROOT / 'main.py'), *args],
                          cwd=tempfile.gettempdir(), text=True, capture_output=True)


class KernelTests(unittest.TestCase):
    def test_files_are_found_by_folder_name(self):
        kernel = flow.find_kernel('matmul')
        self.assertEqual(kernel['top'], 'matmul')
        self.assertEqual(kernel['sources'], [ROOT / 'src/hls/matmul/matmul.cpp'])
        self.assertEqual(kernel['testbench'], [ROOT / 'tb/matmul/test_matmul.cpp'])
        self.assertIn(ROOT / 'src/hls/matmul', kernel['include_dirs'])
        self.assertEqual(kernel['directives'], ROOT / 'config/matmul.tcl')

    def test_config_can_rename_the_top_function(self):
        self.assertEqual(flow.find_kernel('relu_golden')['top'], 'relu_golden_top')

    def test_unknown_or_escaping_kernel_is_rejected(self):
        for name in ('no_such_kernel', '../hls', ''):
            with self.subTest(name=name), self.assertRaisesRegex(SystemExit, 'src/hls/'):
                flow.find_kernel(name)

    def test_kernels_command_lists_them(self):
        result = cli('kernels')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('matmul\n  top        matmul', result.stdout)
        self.assertIn('top        relu_golden_top', result.stdout)

    def test_a_stage_needs_a_kernel(self):
        result = cli('csim')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--kernel', result.stderr)

    def test_old_stages_are_gone(self):
        for stage in ('export', 'synth', 'impl', 'doctor', 'parts'):
            with self.subTest(stage=stage):
                self.assertNotEqual(cli(stage, '--kernel', 'matmul').returncode, 0)


class RunFolderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saved, flow.ROOT = flow.ROOT, pathlib.Path(self.tmp.name)

    def tearDown(self):
        flow.ROOT = self.saved
        self.tmp.cleanup()

    def test_runs_in_the_same_second_get_their_own_folders(self):
        first, second = flow.new_run_folder('k', 'csim'), flow.new_run_folder('k', 'csim')
        self.assertNotEqual(first, second)
        self.assertRegex(first.name, r'^\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d-csim$')
        self.assertTrue(second.name.startswith(first.name))
        self.assertEqual(first.parent, flow.ROOT / 'build' / 'k')

    def test_failing_tool_stops_the_run_and_keeps_its_output(self):
        run = flow.new_run_folder('k', 'csim')
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(SystemExit, 'exit code 7'):
                flow.run_tool([sys.executable, '-c', 'print("broken");exit(7)'], run, 'tool.log')
        self.assertIn('broken', (run / 'tool.log').read_text())


@unittest.skipIf(tkinter is None, 'needs Python tkinter (no display)')
class SettingsTests(unittest.TestCase):
    def test_settings_reach_tcl_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = pathlib.Path(tmp) / 'a folder with spaces'
            run.mkdir()
            flow.write_settings(flow.find_kernel('matmul'), run, 'csynth', 5.0, False)
            tcl = tkinter.Tcl()
            tcl.eval(f'source {{{(run / "settings.tcl").as_posix()}}}')
            self.assertEqual(tcl.getvar('cfg(run_dir)'), run.as_posix())
            self.assertEqual(tcl.getvar('cfg(top)'), 'matmul')
            self.assertEqual(tcl.getvar('cfg(part)'), 'xczu7ev-ffvc1156-2-e')
            self.assertEqual(float(tcl.getvar('cfg(clock_ns)')), 5.0)
            sources = tcl.splitlist(tcl.getvar('cfg(sources)'))
            self.assertEqual(list(sources), [(ROOT / 'src/hls/matmul/matmul.cpp').as_posix()])


class EstimateTests(unittest.TestCase):
    def estimates(self, xml):
        saved = flow.ROOT
        with tempfile.TemporaryDirectory() as tmp:
            flow.ROOT = pathlib.Path(tmp)
            report = flow.ROOT / 'hls/solution/syn/report'
            report.mkdir(parents=True)
            if xml:
                shutil.copy2(xml, report / 'matmul_csynth.xml')
            out = io.StringIO()
            try:
                with contextlib.redirect_stdout(out):
                    flow.print_estimates(flow.ROOT, 'matmul')
            finally:
                flow.ROOT = saved
        return out.getvalue()

    def test_the_numbers_are_printed(self):
        text = self.estimates(ROOT / 'tests/data/matmul_csynth.xml')
        self.assertIn('latency   261 cycles, a new start every 262 cycles', text)
        self.assertIn('clock     4.262 ns needed, 10.00 ns target', text)
        self.assertIn('DSP 8/1728', text)

    def test_no_report_prints_nothing(self):
        self.assertEqual(self.estimates(None), '')


if __name__ == '__main__':
    unittest.main()
