"""Runner regression tests. No AMD tools or FPGA required."""
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('flow', ROOT / 'main.py')
flow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flow)


class FlowTests(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / 'main.py'),
                               *args], cwd=tempfile.gettempdir(),
                              text=True, capture_output=True)

    def config_file(self, directory, **changes):
        """A copy of the project config with fields replaced."""
        path = pathlib.Path(directory) / 'profile.json'
        config = json.loads((ROOT / 'config/project.json').read_text())
        config.update(changes)
        path.write_text(json.dumps(config))
        return path

    def test_hls_dry_run_without_tools(self):
        result = self.cli('cosim', '--dry-run', '--part', 'xc7z020clg400-1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('hls.tcl', result.stdout)
        self.assertIn('cosim', result.stdout)

    def test_bitstream_requires_board(self):
        result = self.cli('bitstream', '--dry-run', '--part', 'xc7z020clg400-1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('board', result.stderr.lower())

    def test_hls_requires_explicit_device(self):
        result = self.cli('csynth', '--dry-run')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('part', result.stderr.lower())

    def test_unknown_kernel_names_the_files_to_create(self):
        result = self.cli('csim', '--kernel', 'missing', '--dry-run')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('src/hls/missing', result.stderr)
        self.assertIn('tb/missing', result.stderr)

    def test_kernel_is_discovered_without_a_config_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.config_file(tmp, kernels={}, part='xc7z020clg400-1')
            result = self.cli('csynth', '--kernel', 'matmul', '--config', str(path), '--dry-run')
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_config_entry_overrides_discovery(self):
        kernel = flow.resolve_kernel('matmul', {'kernels': {'matmul': {'top': 'renamed'}}})
        self.assertEqual(kernel['top'], 'renamed')
        self.assertEqual(kernel['sources'], ['src/hls/matmul/matmul.cpp'])
        self.assertEqual(kernel['directives'], 'config/matmul.tcl')

    def test_kernel_name_cannot_escape_the_layout(self):
        with self.assertRaises(ValueError):
            flow.resolve_kernel('../../etc', {})

    def test_kernels_command_lists_resolved_files(self):
        result = self.cli('kernels')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('matmul', result.stdout)
        self.assertIn('tb/matmul/test_matmul.cpp', result.stdout)

    def test_tcl_strings_roundtrip(self):
        try:
            import tkinter
        except ImportError:
            self.skipTest('Optional Tcl roundtrip test requires Python tkinter (no display needed)')
        tcl = tkinter.Tcl()
        value = 'C:/a space/$x[exit]/brace{here}/semi;colon/"quote"'
        tcl.eval('set value ' + flow.tcl_quote(value))
        self.assertEqual(tcl.getvar('value'), value)

    def test_nonzero_tool_status_is_propagated(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                flow.run_command([sys.executable, '-c', 'print("failed");exit(7)'],
                                 pathlib.Path(tmp), pathlib.Path(tmp) / 'run.log')
            self.assertEqual(caught.exception.returncode, 7)
            self.assertIn('failed', (pathlib.Path(tmp) / 'run.log').read_text())

    def test_selected_compiler_runtime_takes_precedence(self):
        env = flow.compiler_environment(str(ROOT / 'compiler bin' / 'g++'))
        self.assertEqual(env['PATH'].split(os.pathsep)[0], str(ROOT / 'compiler bin'))

    def test_invalid_clock_rejected(self):
        result = self.cli('csynth', '--part', 'xc7z020clg400-1', '--clock-ns', 'nan', '--dry-run')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('positive', result.stderr)

    def test_board_profile_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.config_file(tmp, part='xc7z020clg400-1',
                                    board_script='config/matmul.tcl')
            result = self.cli('bitstream', '--config', str(path), '--dry-run')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('vivado.tcl', result.stdout)


class PartTests(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / 'main.py'),
                               *args], cwd=tempfile.gettempdir(),
                              text=True, capture_output=True)

    def test_named_target_resolves_to_a_part(self):
        self.assertEqual(flow.resolve_part('zcu104'), 'xczu7ev-ffvc1156-2-e')
        self.assertEqual(flow.resolve_part('zybo-z7-20'), 'xc7z020clg400-1')

    def test_exact_part_passes_through(self):
        self.assertEqual(flow.resolve_part('xc7a35ticsg324-1L'), 'xc7a35ticsg324-1L')
        self.assertEqual(flow.resolve_part('anything-else'), 'anything-else')

    def test_every_named_target_looks_like_a_part(self):
        """Guards against typos when someone adds a board."""
        for group, entries in flow.part_groups().items():
            for alias, part in entries.items():
                with self.subTest(target=alias):
                    self.assertRegex(part, r'^xc[a-z0-9]+[a-zA-Z0-9-]*$',
                                     f'{alias} in {group} does not look like an AMD part')
                    self.assertNotIn(' ', part)

    def test_alias_is_used_and_reported(self):
        result = self.cli('csynth', '--part', 'zcu104', '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('xczu7ev-ffvc1156-2-e (zcu104)', result.stdout)

    def test_parts_command_lists_families(self):
        result = self.cli('parts')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('zynq-ultrascale-plus', result.stdout)
        self.assertIn('xc7z020clg400-1', result.stdout)

    def test_implausible_part_warns_but_still_runs(self):
        result = self.cli('csynth', '--part', 'zybo', '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Warning', result.stderr)


class CsynthSummaryTests(unittest.TestCase):
    def summarize(self, tmp):
        report = pathlib.Path(tmp) / 'hls/solution/syn/report'
        report.mkdir(parents=True)
        shutil.copy2(ROOT / 'tests/data/matmul_csynth.xml', report / 'matmul_csynth.xml')
        return flow.summarize_csynth(pathlib.Path(tmp), 'matmul')

    def test_report_fields_are_extracted(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = self.summarize(tmp)
        self.assertEqual(summary['latency_max'], '261')
        self.assertEqual(summary['interval_min'], '262')
        self.assertEqual(summary['clock_estimate'], '4.262')
        self.assertEqual(summary['clock_target'], '10.00')     # lives in UserAssignments
        self.assertEqual(summary['clock_uncertainty'], '2.70')
        self.assertEqual(summary['part'], 'xczu7ev-ffvc1156-2-e')
        self.assertEqual(summary['resources']['DSP'], {'used': '8', 'available': '1728'})

    def test_summary_prints_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            flow.print_csynth_summary(self.summarize(tmp))

    def test_missing_report_is_empty_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(flow.summarize_csynth(pathlib.Path(tmp), 'matmul'), {})

    def test_unrecognized_report_is_empty_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = pathlib.Path(tmp) / 'hls/solution/syn/report'
            report.mkdir(parents=True)
            (report / 'matmul_csynth.xml').write_text('<profile><Other/></profile>')
            self.assertEqual(flow.summarize_csynth(pathlib.Path(tmp), 'matmul'), {})


if __name__ == '__main__':
    unittest.main()
