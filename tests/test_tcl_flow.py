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
        for path in ('scripts/hls.tcl', 'scripts/vivado.tcl', 'boards/zcu104/system.tcl',
                     'boards/zynq7030/system.tcl'):
            self.assertEqual(int(tcl.call('info', 'complete', (ROOT / path).read_text(encoding='utf-8'))), 1)

    def test_every_configured_board_has_its_script(self):
        for board, target in flow.CONFIG['boards'].items():
            with self.subTest(board=board):
                self.assertTrue((ROOT / target['script']).is_file(), target['script'])
        self.assertIn(flow.CONFIG['board'], flow.CONFIG['boards'])

    def wire_zcu104(self, masters):
        return self.wire_board('zcu104', masters)

    def wire_board(self, board, masters, preset=None):
        """Run boards/BOARD/system.tcl with block design commands that only record.
        preset: the text of a boards/BOARD/ps7_preset.tcl to put in the fake repository."""
        with tempfile.TemporaryDirectory() as tmp:
            if preset is not None:
                (Path(tmp) / 'boards' / board).mkdir(parents=True)
                (Path(tmp) / 'boards' / board / 'ps7_preset.tcl').write_text(preset)
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
            tcl.eval('source ' + braced((ROOT / 'boards' / board / 'system.tcl').as_posix()))
            return ([str(w) for w in tcl.splitlist(tcl.getvar('wires'))],
                    [str(p) for p in tcl.splitlist(tcl.getvar('props'))])

    def test_zcu104_gives_each_master_its_own_hp_port(self):
        wires, props = self.wire_zcu104(['m_axi_gmem2', 'm_axi_gmem0', 'm_axi_gmem1'])  # any order
        for wire in ('ps/M_AXI_HPM0_FPD => ctrl/S00_AXI', 'ctrl/M00_AXI => kernel/s_axi_control',
                     'kernel/interrupt -> ps/pl_ps_irq0', 'ps/pl_resetn0 -> rst/ext_reset_in'):
            self.assertIn(wire, wires)
        for i in range(3):   # m_axi_gmem<i> -> data<i> -> HP<i>, in name order
            self.assertIn(f'kernel/m_axi_gmem{i} => data{i}/S00_AXI', wires)
            self.assertIn(f'data{i}/M00_AXI => ps/S_AXI_HP{i}_FPD', wires)
            self.assertIn(f'ps/pl_clk0 -> data{i}/aclk', wires)
            self.assertIn(f'ps/pl_clk0 -> ps/saxihp{i}_fpd_aclk', wires)
            self.assertIn(f'rst/peripheral_aresetn -> data{i}/aresetn', wires)
        for pin in ('rst/slowest_sync_clk', 'kernel/ap_clk', 'ctrl/aclk', 'ps/maxihpm0_fpd_aclk'):
            self.assertIn(f'ps/pl_clk0 -> {pin}', wires)
        for pin in ('kernel/ap_rst_n', 'ctrl/aresetn'):
            self.assertIn(f'rst/peripheral_aresetn -> {pin}', wires)
        self.assertFalse([w for w in wires if 'HP3' in w or 'saxihp3' in w], wires)
        hp_used = [props[props.index(f'CONFIG.PSU__USE__S_AXI_GP{gp}') + 1] for gp in (2, 3, 4, 5)]
        self.assertEqual(hp_used, ['1', '1', '1', '0'])
        smartconnect_inputs = [props[i + 1] for i, p in enumerate(props) if p == 'CONFIG.NUM_SI']
        self.assertEqual(smartconnect_inputs, ['1'] * 4, 'ctrl, then one per master')

    def test_zcu104_rejects_more_masters_than_hp_ports(self):
        with self.assertRaisesRegex(tkinter.TclError, '4 HP ports'):
            self.wire_zcu104([f'm_axi_gmem{i}' for i in range(5)])

    def test_zcu104_without_masters_has_no_data_path(self):
        wires, props = self.wire_zcu104([])
        self.assertIn('ctrl/M00_AXI => kernel/s_axi_control', wires)
        self.assertFalse([w for w in wires if 'data' in w or 'S_AXI_HP' in w or 'saxihp' in w], wires)
        hp_used = [props[props.index(f'CONFIG.PSU__USE__S_AXI_GP{gp}') + 1] for gp in (2, 3, 4, 5)]
        self.assertEqual(hp_used, ['0'] * 4)

    def test_zynq7030_gives_each_master_its_own_64_bit_hp_port(self):
        wires, props = self.wire_board('zynq7030', ['m_axi_gmem1', 'm_axi_gmem0', 'm_axi_gmem2'])
        for wire in ('ps/M_AXI_GP0 => ctrl/S00_AXI', 'ctrl/M00_AXI => kernel/s_axi_control',
                     'kernel/interrupt -> ps/IRQ_F2P', 'ps/FCLK_RESET0_N -> rst/ext_reset_in'):
            self.assertIn(wire, wires)
        for i in range(3):   # m_axi_gmem<i> -> data<i> -> HP<i>, in name order
            self.assertIn(f'kernel/m_axi_gmem{i} => data{i}/S00_AXI', wires)
            self.assertIn(f'data{i}/M00_AXI => ps/S_AXI_HP{i}', wires)
            self.assertIn(f'ps/FCLK_CLK0 -> ps/S_AXI_HP{i}_ACLK', wires)
            self.assertIn(f'ps/FCLK_CLK0 -> data{i}/aclk', wires)
            self.assertEqual(props[props.index(f'CONFIG.PCW_S_AXI_HP{i}_DATA_WIDTH') + 1], '64')
        for pin in ('rst/slowest_sync_clk', 'kernel/ap_clk', 'ctrl/aclk', 'ps/M_AXI_GP0_ACLK'):
            self.assertIn(f'ps/FCLK_CLK0 -> {pin}', wires)
        hp_used = [props[props.index(f'CONFIG.PCW_USE_S_AXI_HP{i}') + 1] for i in range(4)]
        self.assertEqual(hp_used, ['1', '1', '1', '0'])
        self.assertEqual(props[props.index('CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ') + 1], '100.000')

    def test_zynq7030_applies_the_ps_preset_when_there_is_one(self):
        _, props = self.wire_board('zynq7030', [], preset='set_property -dict [list CONFIG.PCW_X 7] $ps\n')
        self.assertEqual(props[props.index('CONFIG.PCW_X') + 1], '7')
        _, props = self.wire_board('zynq7030', [])
        self.assertNotIn('CONFIG.PCW_X', props)

    def test_zynq7030_rejects_more_masters_than_hp_ports(self):
        with self.assertRaisesRegex(tkinter.TclError, '4 HP ports'):
            self.wire_board('zynq7030', [f'm_axi_gmem{i}' for i in range(5)])

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
