"""Exercise Tcl stage orchestration with AMD commands replaced by recorders.

These check our control flow, not AMD command availability or hardware results.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

try:
    import tkinter
except ImportError:
    tkinter = None

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('flow', ROOT / 'main.py')
flow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flow)


@unittest.skipIf(tkinter is None, 'Optional Tcl tests require Python tkinter; no display needed')
class TclFlowTests(unittest.TestCase):
    def execute_hls(self, stage, fail_at='', skip_cosim=False):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            config = json.loads((ROOT / 'config/project.json').read_text())
            config['part'] = 'xc7z020clg400-1'
            config['skip_cosim'] = skip_cosim
            flow.write_settings(run, config, flow.resolve_kernel('matmul', config), stage)
            tcl = tkinter.Tcl()
            tcl.eval('set calls {}; set exit_code -1')
            for name in ('open_project', 'set_top', 'add_files', 'open_solution',
                         'set_part', 'create_clock', 'csim_design', 'csynth_design',
                         'cosim_design', 'export_design', 'close_project'):
                body = f'lappend ::calls {name}'
                if name == fail_at:
                    body += '; error "injected tool failure"'
                tcl.eval(f'proc {name} {{args}} {{{body}}}')
            tcl.eval('proc exit {code} {set ::exit_code $code; error "test exit"}')
            script = (ROOT / 'scripts/hls.tcl').read_text().replace(
                'source settings.tcl', 'source ' + flow.tcl_quote((run / 'settings.tcl').as_posix()))
            with self.assertRaises(tkinter.TclError):
                tcl.eval(script)
            calls = tcl.splitlist(tcl.getvar('calls'))
            return int(tcl.getvar('exit_code')), calls

    def test_stage_dependencies(self):
        expected = {
            'csim': ['csim_design'],
            'csynth': ['csim_design', 'csynth_design'],
            'cosim': ['csim_design', 'csynth_design', 'cosim_design'],
        }
        for stage in ('export', 'synth', 'impl', 'bitstream'):
            expected[stage] = ['csim_design', 'csynth_design', 'cosim_design', 'export_design']
        for stage, stages in expected.items():
            with self.subTest(stage=stage):
                code, calls = self.execute_hls(stage)
                self.assertEqual(code, 0)
                self.assertEqual([c for c in calls if c.endswith('_design')], stages)

    def test_skip_cosim_still_exports(self):
        for stage in ('export', 'bitstream'):
            with self.subTest(stage=stage):
                code, calls = self.execute_hls(stage, skip_cosim=True)
                self.assertEqual(code, 0)
                self.assertEqual([c for c in calls if c.endswith('_design')],
                                 ['csim_design', 'csynth_design', 'export_design'])

    def test_simulation_failure_stops_export(self):
        code, calls = self.execute_hls('export', fail_at='csim_design')
        self.assertEqual(code, 1)
        self.assertNotIn('csynth_design', calls)
        self.assertNotIn('export_design', calls)

    def test_scripts_have_complete_tcl_syntax(self):
        tcl = tkinter.Tcl()
        for name in ('hls.tcl', 'vivado.tcl'):
            self.assertEqual(int(tcl.call('info', 'complete', (ROOT / 'scripts' / name).read_text())), 1)

    def test_kernel_clock_is_loaded_before_vivado_synthesis(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            rtl = run / 'hls/solution/syn/verilog'
            rtl.mkdir(parents=True)
            (rtl / 'matmul.v').write_text('module matmul(input ap_clk); endmodule')
            config = json.loads((ROOT / 'config/project.json').read_text())
            config['part'] = 'xc7z020clg400-1'
            flow.write_settings(run, config, flow.resolve_kernel('matmul', config), 'synth')
            tcl = tkinter.Tcl()
            tcl.eval('set calls {}; set exit_code -1')
            for name in ('set_param', 'create_project', 'add_files', 'synth_design',
                         'create_clock', 'get_ports', 'report_utilization', 'read_xdc',
                         'report_timing_summary', 'report_drc', 'report_power', 'write_checkpoint'):
                tcl.eval(f'proc {name} {{args}} {{lappend ::calls {name}}}')
            tcl.eval('proc exit {code} {set ::exit_code $code; error "test exit"}')
            script = (ROOT / 'scripts/vivado.tcl').read_text().replace(
                'source settings.tcl', 'source ' + flow.tcl_quote((run / 'settings.tcl').as_posix()))
            with self.assertRaises(tkinter.TclError):
                tcl.eval(script)
            self.assertEqual(int(tcl.getvar('exit_code')), 0)
            calls = tcl.splitlist(tcl.getvar('calls'))
            self.assertIn('read_xdc', calls, 'Synthesis needs the timing XDC loaded first')
            self.assertLess(calls.index('read_xdc'), calls.index('synth_design'))
            clock_file = run / 'kernel_clock.xdc'
            self.assertIn('[get_ports ap_clk]', clock_file.read_text())
            self.assertIn('-period 10.0', clock_file.read_text())


if __name__ == '__main__':
    unittest.main()
