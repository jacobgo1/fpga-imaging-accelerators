"""Exercise Tcl stage orchestration with AMD commands replaced by recorders.

These check our control flow, not AMD command availability or hardware results.
"""
import importlib.util
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


def braced(path):
    return '{' + path + '}'


@unittest.skipIf(tkinter is None, 'Optional Tcl tests require Python tkinter; no display needed')
class TclFlowTests(unittest.TestCase):
    def execute_hls(self, stage, fail_at='', skip_cosim=False, csynth_report=True):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            flow.write_settings(flow.find_kernel('matmul'), run, stage, 10.0, skip_cosim)
            tcl = tkinter.Tcl()
            tcl.eval('set calls {}; set exit_code -1')
            for name in ('open_project', 'set_top', 'add_files', 'open_solution',
                         'set_part', 'create_clock', 'csim_design', 'csynth_design',
                         'cosim_design', 'export_design', 'close_project'):
                body = f'lappend ::calls {name}'
                if name == 'csynth_design' and csynth_report:  # what a real run leaves behind
                    body += ('; set d [file join $::cfg(run_dir) hls solution syn report]'
                             '; file mkdir $d; close [open [file join $d $::cfg(top)_csynth.rpt] w]')
                if name == fail_at:
                    body += '; error "injected tool failure"'
                tcl.eval(f'proc {name} {{args}} {{{body}}}')
            tcl.eval('proc exit {code} {set ::exit_code $code; error "test exit"}')
            script = (ROOT / 'scripts/hls.tcl').read_text().replace(
                'source settings.tcl', 'source ' + braced((run / 'settings.tcl').as_posix()))
            with self.assertRaises(tkinter.TclError):
                tcl.eval(script)
            calls = tcl.splitlist(tcl.getvar('calls'))
            return int(tcl.getvar('exit_code')), calls

    def test_stage_dependencies(self):
        expected = {
            'csim': ['csim_design'],
            'csynth': ['csim_design', 'csynth_design'],
            'cosim': ['csim_design', 'csynth_design', 'cosim_design'],
            'bitstream': ['csim_design', 'csynth_design', 'cosim_design', 'export_design'],
        }
        for stage, stages in expected.items():
            with self.subTest(stage=stage):
                code, calls = self.execute_hls(stage)
                self.assertEqual(code, 0)
                self.assertEqual([c for c in calls if c.endswith('_design')], stages)

    def test_skip_cosim_still_exports(self):
        code, calls = self.execute_hls('bitstream', skip_cosim=True)
        self.assertEqual(code, 0)
        self.assertEqual([c for c in calls if c.endswith('_design')],
                         ['csim_design', 'csynth_design', 'export_design'])

    def test_simulation_failure_stops_export(self):
        code, calls = self.execute_hls('bitstream', fail_at='csim_design')
        self.assertEqual(code, 1)
        self.assertNotIn('csynth_design', calls)
        self.assertNotIn('export_design', calls)

    def test_synthesis_that_fails_without_raising_stops_export(self):
        # Vitis HLS reports some failures ("Pre-synthesis failed") only as a
        # message and returns normally: no report must mean no export.
        code, calls = self.execute_hls('bitstream', skip_cosim=True, csynth_report=False)
        self.assertEqual(code, 1)
        self.assertIn('csynth_design', calls)
        self.assertNotIn('export_design', calls)

    def test_scripts_have_complete_tcl_syntax(self):
        tcl = tkinter.Tcl()
        for path in ('scripts/hls.tcl', 'scripts/vivado.tcl', 'boards/zcu104/system.tcl'):
            self.assertEqual(int(tcl.call('info', 'complete', (ROOT / path).read_text())), 1)

    def wire_zcu104(self, masters):
        """Run boards/zcu104/system.tcl with block design commands that only record."""
        with tempfile.TemporaryDirectory() as tmp:
            tcl = tkinter.Tcl()
            tcl.eval(f'set cfg(top) k; set cfg(clock_ns) 10.0; set cfg(run_dir) {{{tmp}}}; '
                     f'set cfg(root) {{{tmp}}}; set ip_repo {{{tmp}}}; set wires {{}}; set props {{}}')
            tcl.eval(f'set masters {{{" ".join("kernel/" + m for m in masters)}}}')
            for name in ('set_property', 'current_project', 'create_bd_design', 'apply_bd_automation',
                         'assign_bd_address', 'validate_bd_design', 'save_bd_design', 'get_files',
                         'generate_target', 'add_files', 'make_wrapper', 'get_filesets',
                         'update_compile_order', 'get_bd_addr_segs', 'get_bd_addr_spaces'):
                tcl.eval(f'proc {name} {{args}} {{}}')
            tcl.eval('proc set_property {args} {'
                     ' if {[lindex $args 0] eq "-dict"} {lappend ::props {*}[lindex $args 1]}}')
            tcl.eval('proc get_board_parts {args} {return xilinx.com:zcu104:part0:1.1}')
            tcl.eval('proc get_ipdefs {args} {return xilinx.com:hls:k:1.0}')
            tcl.eval('proc create_bd_cell {args} {return [lindex $args end]}')
            tcl.eval('proc get_property {args} {return 100.0}')
            tcl.eval('proc get_bd_pins {args} {return [lindex $args end]}')
            tcl.eval('proc get_bd_intf_pins {args} {'
                     ' if {"-of" in $args} {return $::masters}; return [lindex $args end]}')
            tcl.eval('proc connect_bd_net {a b} {lappend ::wires "$a -> $b"}')
            tcl.eval('proc connect_bd_intf_net {a b} {lappend ::wires "$a => $b"}')
            tcl.eval('source ' + braced((ROOT / 'boards/zcu104/system.tcl').as_posix()))
            return ([str(w) for w in tcl.splitlist(tcl.getvar('wires'))],
                    [str(p) for p in tcl.splitlist(tcl.getvar('props'))])

    def test_zcu104_wires_two_masters_through_one_smartconnect_to_hp0(self):
        wires, props = self.wire_zcu104(['m_axi_gmem0', 'm_axi_gmem1'])
        for wire in ('ps/M_AXI_HPM0_FPD => ctrl/S00_AXI', 'ctrl/M00_AXI => kernel/s_axi_control',
                     'kernel/m_axi_gmem0 => data/S00_AXI', 'kernel/m_axi_gmem1 => data/S01_AXI',
                     'data/M00_AXI => ps/S_AXI_HP0_FPD', 'kernel/interrupt -> ps/pl_ps_irq0',
                     'ps/pl_resetn0 -> rst/ext_reset_in'):
            self.assertIn(wire, wires)
        for pin in ('rst/slowest_sync_clk', 'kernel/ap_clk', 'ctrl/aclk', 'data/aclk',
                    'ps/maxihpm0_fpd_aclk', 'ps/saxihp0_fpd_aclk'):
            self.assertIn(f'ps/pl_clk0 -> {pin}', wires)
        for pin in ('kernel/ap_rst_n', 'ctrl/aresetn', 'data/aresetn'):
            self.assertIn(f'rst/peripheral_aresetn -> {pin}', wires)
        self.assertEqual(props[props.index('CONFIG.PSU__USE__S_AXI_GP2') + 1], '1')
        self.assertEqual(props[props.index('CONFIG.NUM_SI', props.index('CONFIG.NUM_SI') + 1) + 1], '2')

    def test_zcu104_without_masters_has_no_data_path(self):
        wires, props = self.wire_zcu104([])
        self.assertIn('ctrl/M00_AXI => kernel/s_axi_control', wires)
        self.assertFalse([w for w in wires if 'data/' in w or 'HP0' in w or 'saxihp0' in w], wires)
        self.assertEqual(props[props.index('CONFIG.PSU__USE__S_AXI_GP2') + 1], '0')

    def execute_vivado(self, progress='100%'):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            flow.write_settings(flow.find_kernel('matmul'), run, 'bitstream', 10.0, True)
            tcl = tkinter.Tcl()
            tcl.eval('set calls {}; set exit_code -1')
            for name in ('set_param', 'create_project', 'set_property', 'current_project', 'update_ip_catalog',
                         'launch_runs', 'wait_on_run', 'get_runs', 'open_run', 'report_utilization',
                         'report_timing_summary', 'report_drc', 'report_power'):
                tcl.eval(f'proc {name} {{args}} {{lappend ::calls [list {name} {{*}}$args]}}')
            tcl.eval('proc get_timing_paths {args} {return {}}')  # timing met
            tcl.eval(f'proc get_property {{name args}} {{if {{$name eq "PROGRESS"}} {{return {progress}}}; '
                     f'return {{{run.as_posix()}}}}}')
            (run / 'system_wrapper.bit').write_bytes(b'')
            tcl.eval('proc exit {code} {set ::exit_code $code; error "test exit"}')
            script = (ROOT / 'scripts/vivado.tcl').read_text().replace(
                'source settings.tcl', 'source ' + braced((run / 'settings.tcl').as_posix()) +
                '; set cfg(board_script) [file join $cfg(run_dir) board.tcl]')
            (run / 'board.tcl').write_text('lappend ::calls board_script\n')
            with self.assertRaises(tkinter.TclError):
                tcl.eval(script)
            return int(tcl.getvar('exit_code')), [tcl.splitlist(c)[0] for c in tcl.splitlist(tcl.getvar('calls'))]

    def test_vivado_builds_the_design_then_synthesizes_then_implements(self):
        code, calls = self.execute_vivado()
        self.assertEqual(code, 0)
        steps = [c for c in calls if c in ('create_project', 'board_script', 'launch_runs')]
        self.assertEqual(steps, ['create_project', 'board_script', 'launch_runs', 'launch_runs'])

    def test_failed_synthesis_stops_vivado(self):
        code, calls = self.execute_vivado(progress='0%')
        self.assertEqual(code, 1)
        self.assertEqual(calls.count('launch_runs'), 1)

if __name__ == '__main__':
    unittest.main()
