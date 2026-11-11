"""Exercise Tcl stage orchestration with AMD commands replaced by recorders.

These check our control flow, not AMD command availability or hardware results.
"""
from pathlib import Path
import tempfile
import unittest

from helpers import ROOT

try:
    import tkinter
except ImportError:
    tkinter = None

KERNEL_CPP = ROOT / 'src/models/2D-Justo-UNet-Simple/justounetsimple_golden/justounetsimple.cpp'


def braced(path):
    return '{' + path + '}'


def write_hls_settings(run, stage, cosim=False, profile=False):
    """settings.tcl as scripts/hls.sh writes it (tests/test_scripts.py checks that)."""
    (run / 'settings.tcl').write_text(
        f'set cfg(run_dir) {braced(run.as_posix())}\nset cfg(stage) {stage}\nset cfg(top) justounetsimple\n'
        f'set cfg(part) xczu7ev-ffvc1156-2-e\nset cfg(clock_ns) 10.0\nset cfg(cosim) {int(cosim)}\n'
        f'set cfg(profile) {int(profile)}\nset cfg(sources) [list {braced(KERNEL_CPP.as_posix())}]\n'
        f'set cfg(testbench) [list]\nset cfg(include_dirs) [list {braced(KERNEL_CPP.parent.as_posix())}]\n')


@unittest.skipIf(tkinter is None, 'Optional Tcl tests require Python tkinter; no display needed')
class TclFlowTests(unittest.TestCase):
    def execute_hls(self, stage, fail_at='', cosim=True, csynth_report=True, profile=False):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_hls_settings(run, stage, cosim or stage == 'cosim', profile)
            tcl = tkinter.Tcl()
            tcl.eval('set calls {}; set exit_code -1; set cosim_args {}')
            for name in ('open_project', 'set_top', 'add_files', 'open_solution',
                         'set_part', 'create_clock', 'csim_design', 'csynth_design',
                         'cosim_design', 'export_design', 'close_project'):
                body = f'lappend ::calls {name}'
                if name == 'csynth_design' and csynth_report:  # what a real run leaves behind
                    body += ('; set d [file join $::cfg(run_dir) hls solution syn report]'
                             '; file mkdir $d; close [open [file join $d $::cfg(top)_csynth.rpt] w]')
                if name == 'cosim_design':
                    body += '; set ::cosim_args $args'
                if name == fail_at:
                    body += '; error "injected tool failure"'
                tcl.eval(f'proc {name} {{args}} {{{body}}}')
            tcl.eval('proc exit {code} {set ::exit_code $code; error "test exit"}')
            script = (ROOT / 'scripts/hls.tcl').read_text().replace(
                'source settings.tcl', 'source ' + braced((run / 'settings.tcl').as_posix()))
            with self.assertRaises(tkinter.TclError):
                tcl.eval(script)
            calls = tcl.splitlist(tcl.getvar('calls'))
            self.cosim_args = [str(a) for a in tcl.splitlist(tcl.getvar('cosim_args'))]
            return int(tcl.getvar('exit_code')), calls

    def test_stage_dependencies(self):
        expected = {
            'csim': ['csim_design'],
            'csynth': ['csim_design', 'csynth_design'],
            'cosim': ['csim_design', 'csynth_design', 'cosim_design'],
            'ip': ['csim_design', 'csynth_design', 'cosim_design', 'export_design'],
        }
        for stage, stages in expected.items():
            with self.subTest(stage=stage):
                code, calls = self.execute_hls(stage)
                self.assertEqual(code, 0)
                self.assertEqual([c for c in calls if c.endswith('_design')], stages)

    def test_dataflow_profile_records_what_the_dataflow_viewer_needs(self):
        self.execute_hls('cosim')
        self.assertNotIn('-enable_dataflow_profiling', self.cosim_args)
        self.assertEqual(self.cosim_args[self.cosim_args.index('-trace_level') + 1], 'port')
        self.execute_hls('cosim', profile=True)
        self.assertIn('-enable_dataflow_profiling', self.cosim_args)
        self.assertEqual(self.cosim_args[self.cosim_args.index('-trace_level') + 1], 'all')

    def test_ip_without_cosim_still_exports(self):
        code, calls = self.execute_hls('ip', cosim=False)
        self.assertEqual(code, 0)
        self.assertEqual([c for c in calls if c.endswith('_design')],
                         ['csim_design', 'csynth_design', 'export_design'])

    def test_simulation_failure_stops_export(self):
        code, calls = self.execute_hls('ip', fail_at='csim_design')
        self.assertEqual(code, 1)
        self.assertNotIn('csynth_design', calls)
        self.assertNotIn('export_design', calls)

    def test_synthesis_that_fails_without_raising_stops_export(self):
        # Vitis HLS reports some failures ("Pre-synthesis failed") only as a
        # message and returns normally: no report must mean no export.
        code, calls = self.execute_hls('ip', cosim=False, csynth_report=False)
        self.assertEqual(code, 1)
        self.assertIn('csynth_design', calls)
        self.assertNotIn('export_design', calls)

    def test_scripts_have_complete_tcl_syntax(self):
        tcl = tkinter.Tcl()
        for path in ('scripts/hls.tcl', 'scripts/vivado.tcl', 'boards/zcu104/system.tcl',
                     'boards/pynqz2/system.tcl', 'boards/opu/project.tcl', 'boards/opu/add_hls_kernel.tcl'):
            with self.subTest(path=path):
                self.assertEqual(int(tcl.call('info', 'complete', (ROOT / path).read_text(encoding='utf-8'))), 1)

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
            tcl.eval(f'set cfg(top) k; set cfg(board) {board}; set cfg(clock_ns) 10.0; set cfg(run_dir) {{{tmp}}}; '
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

    def test_zcu104_puts_the_fifth_and_sixth_master_on_the_hpc_ports(self):
        wires, props = self.wire_zcu104([f'm_axi_gmem{i}' for i in range(6)])
        for i, port in enumerate(('HP0', 'HP1', 'HP2', 'HP3', 'HPC0', 'HPC1')):
            self.assertIn(f'kernel/m_axi_gmem{i} => data{i}/S00_AXI', wires)
            self.assertIn(f'data{i}/M00_AXI => ps/S_AXI_{port}_FPD', wires)
            self.assertIn(f'ps/pl_clk0 -> ps/saxi{port.lower()}_fpd_aclk', wires)
        used = [props[props.index(f'CONFIG.PSU__USE__S_AXI_GP{gp}') + 1] for gp in (2, 3, 4, 5, 0, 1)]
        self.assertEqual(used, ['1'] * 6)

    def test_zcu104_rejects_more_masters_than_ports_to_ddr(self):
        with self.assertRaisesRegex(tkinter.TclError, '6 ports to DDR'):
            self.wire_zcu104([f'm_axi_gmem{i}' for i in range(7)])

    def test_zcu104_without_masters_has_no_data_path(self):
        wires, props = self.wire_zcu104([])
        self.assertIn('ctrl/M00_AXI => kernel/s_axi_control', wires)
        self.assertFalse([w for w in wires if 'data' in w or 'S_AXI_HP' in w or 'saxihp' in w], wires)
        hp_used = [props[props.index(f'CONFIG.PSU__USE__S_AXI_GP{gp}') + 1] for gp in (2, 3, 4, 5)]
        self.assertEqual(hp_used, ['0'] * 4)

    def test_pynqz2_gives_each_master_its_own_64_bit_hp_port(self):
        wires, props = self.wire_board('pynqz2', ['m_axi_gmem1', 'm_axi_gmem0', 'm_axi_gmem2'])
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

    def test_pynqz2_applies_the_ps_preset_when_there_is_one(self):
        _, props = self.wire_board('pynqz2', [], preset='set_property -dict [list CONFIG.PCW_X 7] $ps\n')
        self.assertEqual(props[props.index('CONFIG.PCW_X') + 1], '7')
        _, props = self.wire_board('pynqz2', [])
        self.assertNotIn('CONFIG.PCW_X', props)

    def test_pynqz2_puts_a_fifth_master_on_hp3_with_the_fourth(self):
        wires, props = self.wire_board('pynqz2', [f'm_axi_gmem{i}' for i in range(5)])
        for i in range(4):
            self.assertIn(f'kernel/m_axi_gmem{i} => data{i}/S00_AXI', wires)
            self.assertIn(f'data{i}/M00_AXI => ps/S_AXI_HP{i}', wires)
        self.assertIn('kernel/m_axi_gmem4 => data3/S01_AXI', wires)
        self.assertEqual([w for w in wires if w.endswith('S_AXI_HP3')], ['data3/M00_AXI => ps/S_AXI_HP3'])
        smartconnect_inputs = [props[i + 1] for i, p in enumerate(props) if p == 'CONFIG.NUM_SI']
        self.assertEqual(smartconnect_inputs, ['1', '1', '1', '1', '2'], 'ctrl, then data0-3')
        hp_used = [props[props.index(f'CONFIG.PCW_USE_S_AXI_HP{i}') + 1] for i in range(4)]
        self.assertEqual(hp_used, ['1'] * 4)

    def test_pynqz2_rejects_more_masters_than_hp_ports(self):
        with self.assertRaisesRegex(tkinter.TclError, '4 HP ports'):
            self.wire_board('pynqz2', [f'm_axi_gmem{i}' for i in range(6)])

    def add_to_opu(self, masters, clock_ns=10.0, fclk_mhz=100.0):
        """Run boards/opu/add_hls_kernel.tcl on a stand-in of opu-system's base design, with
        block design commands that only record. Returns the wires, the properties set,
        the blocks made, the pins taken off their nets, and the post-route hook."""
        with tempfile.TemporaryDirectory() as tmp:
            tcl = tkinter.Tcl()
            tcl.eval(f'set projectDir {{{tmp}}}; set projectName p; set hlsTop k; set hlsClockNs {clock_ns}; '
                     f'set fclk {fclk_mhz}; set wires {{}}; set props {{}}; set cells {{}}; set cut {{}}')
            tcl.eval(f'set masters {{{" ".join("hls_kernel/" + m for m in masters)}}}')
            for name in ('current_project', 'update_ip_catalog', 'assign_bd_address', 'get_runs'):
                tcl.eval(f'proc {name} {{args}} {{return [lindex $args end]}}')
            tcl.eval('proc set_property {args} {'
                     ' if {[lindex $args 0] eq "-dict"} {lappend ::props {*}[lindex $args 1]} '
                     ' else {lappend ::props [lindex $args 0] [lindex $args 1]}}')
            tcl.eval('proc get_property {name args} {switch -glob -- $name {'
                     ' CONFIG.PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ {return $::fclk} CONFIG.NUM_MI {return 3}'
                     ' ip_repo_paths {return cubedma_7030} NAME {return [file tail [lindex $args 0]]}}}')
            tcl.eval('proc get_ipdefs {args} {return xilinx.com:hls:k:1.0}')
            tcl.eval('proc create_bd_cell {args} {lappend ::cells [lindex $args end]; return [lindex $args end]}')
            tcl.eval('proc get_bd_cells {args} {return [lindex $args end]}')
            tcl.eval('proc get_bd_pins {args} {return [lindex $args end]}')
            tcl.eval('proc get_bd_intf_pins {args} {'
                     ' if {"-of" in $args} {return $::masters}; return [lindex $args end]}')
            # opu_base.tcl ties HP0's clock to the GP0 clock's net.
            tcl.eval('proc get_bd_nets {args} {'
                     ' if {[lindex $args end] eq "processing_system7_0/S_AXI_HP0_ACLK"} {return gp0_clk_net}; return {}}')
            tcl.eval('proc disconnect_bd_net {net pin} {lappend ::cut "$net $pin"}')
            tcl.eval('proc connect_bd_net {a b} {lappend ::wires "$a -> $b"}')
            tcl.eval('proc connect_bd_intf_net {a b} {lappend ::wires "$a => $b"}')
            tcl.eval('source ' + braced((ROOT / 'boards/opu/add_hls_kernel.tcl').as_posix()))
            hook = Path(tmp) / 'p' / 'hls_reports.tcl'
            return ([str(w) for w in tcl.splitlist(tcl.getvar('wires'))],
                    [str(p) for p in tcl.splitlist(tcl.getvar('props'))],
                    [str(c) for c in tcl.splitlist(tcl.getvar('cells'))],
                    [str(c) for c in tcl.splitlist(tcl.getvar('cut'))],
                    hook.read_text() if hook.exists() else None)

    def test_opu_kernel_at_the_boards_clock_shares_fclk0_and_the_base_reset(self):
        wires, props, cells, cut, hook = self.add_to_opu(['m_axi_gmem1', 'm_axi_gmem0', 'm_axi_gmem2'])
        self.assertNotIn('hls_clock', cells)
        clk, rstn = 'processing_system7_0/FCLK_CLK0', 'proc_sys_reset_0/peripheral_aresetn'
        for wire in (f'{clk} -> hls_kernel/ap_clk', f'{rstn} -> hls_kernel/ap_rst_n',
                     'axi_interconnect_0/M03_AXI => hls_kernel/s_axi_control',
                     f'{clk} -> axi_interconnect_0/M03_ACLK', f'{rstn} -> axi_interconnect_0/M03_ARESETN'):
            self.assertIn(wire, wires)
        self.assertEqual(props[props.index('CONFIG.NUM_MI') + 1], '4', 'one more master on the interconnect')
        for i in range(3):   # m_axi_gmem<i> -> hls_data<i> -> HP<i>, in name order
            self.assertIn(f'hls_kernel/m_axi_gmem{i} => hls_data{i}/S00_AXI', wires)
            self.assertIn(f'hls_data{i}/M00_AXI => processing_system7_0/S_AXI_HP{i}', wires)
            self.assertIn(f'{clk} -> hls_data{i}/aclk', wires)
            self.assertIn(f'{clk} -> processing_system7_0/S_AXI_HP{i}_ACLK', wires)
            self.assertEqual(props[props.index(f'CONFIG.PCW_USE_S_AXI_HP{i}') + 1], '1')
            self.assertEqual(props[props.index(f'CONFIG.PCW_S_AXI_HP{i}_DATA_WIDTH') + 1], '64')
        self.assertNotIn('CONFIG.PCW_USE_S_AXI_HP3', props)
        self.assertEqual(cut, ['gp0_clk_net processing_system7_0/S_AXI_HP0_ACLK'])
        # The kernel's IP next to the base design's, and the reports after routing.
        self.assertEqual(props[props.index('ip_repo_paths') + 1].split()[0], 'cubedma_7030')
        self.assertIn('report_utilization -hierarchical', hook)
        self.assertIn('STEPS.ROUTE_DESIGN.TCL.POST', props)

    def test_opu_kernel_built_for_a_slower_clock_gets_an_mmcm(self):
        wires, props, cells, _, _ = self.add_to_opu(['m_axi_gmem0', 'm_axi_gmem1'], clock_ns=12.8)
        self.assertIn('hls_clock', cells)
        self.assertIn('hls_reset', cells)
        self.assertEqual(props[props.index('CONFIG.CLKOUT1_REQUESTED_OUT_FREQ') + 1], '78.125')
        clk, rstn = 'hls_clock/clk_out1', 'hls_reset/peripheral_aresetn'
        for wire in ('processing_system7_0/FCLK_CLK0 -> hls_clock/clk_in1',
                     'processing_system7_0/FCLK_RESET0_N -> hls_clock/resetn',
                     f'{clk} -> hls_reset/slowest_sync_clk', 'hls_clock/locked -> hls_reset/dcm_locked',
                     f'{clk} -> hls_kernel/ap_clk', f'{rstn} -> hls_kernel/ap_rst_n',
                     f'{clk} -> axi_interconnect_0/M03_ACLK', f'{rstn} -> axi_interconnect_0/M03_ARESETN',
                     f'{clk} -> hls_data0/aclk', f'{clk} -> processing_system7_0/S_AXI_HP0_ACLK',
                     f'{clk} -> processing_system7_0/S_AXI_HP1_ACLK'):
            self.assertIn(wire, wires)
        self.assertFalse([w for w in wires if w.startswith('processing_system7_0/FCLK_CLK0 -> hls_kernel')])

    def test_opu_puts_a_fifth_master_on_hp3_and_rejects_a_sixth(self):
        wires, props, _, _, _ = self.add_to_opu([f'm_axi_gmem{i}' for i in range(5)])
        self.assertIn('hls_kernel/m_axi_gmem4 => hls_data3/S01_AXI', wires)
        self.assertEqual([props[i + 1] for i, p in enumerate(props) if p == 'CONFIG.NUM_SI'], ['1', '1', '1', '2'])
        with self.assertRaisesRegex(tkinter.TclError, '4 HP ports'):
            self.add_to_opu([f'm_axi_gmem{i}' for i in range(6)])

    def execute_vivado(self, progress='100%'):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'vivado-settings.tcl').write_text(
                f'set cfg(run_dir) {braced(run.as_posix())}\nset cfg(ip_repo) {braced(run.as_posix())}\n'
                'set cfg(top) justounetsimple\nset cfg(part) xczu7ev-ffvc1156-2-e\nset cfg(jobs) 5\n'
                f'set cfg(board_script) {braced((run / "board.tcl").as_posix())}\n')
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
                'source vivado-settings.tcl', 'source ' + braced((run / 'vivado-settings.tcl').as_posix()))
            (run / 'board.tcl').write_text('lappend ::calls board_script\n')
            with self.assertRaises(tkinter.TclError):
                tcl.eval(script)
            calls = [list(map(str, tcl.splitlist(c))) for c in tcl.splitlist(tcl.getvar('calls'))]
            self.threads_tcl = (run / 'threads.tcl').read_text() if (run / 'threads.tcl').exists() else None
            self.vivado_calls = calls
            return int(tcl.getvar('exit_code')), [c[0] for c in calls]

    def test_vivado_builds_the_design_then_synthesizes_then_implements(self):
        code, calls = self.execute_vivado()
        self.assertEqual(code, 0)
        steps = [c for c in calls if c in ('create_project', 'board_script', 'launch_runs')]
        self.assertEqual(steps, ['create_project', 'board_script', 'launch_runs', 'launch_runs'])

    def test_vivado_runs_get_the_thread_count(self):
        """launch_runs starts new Vivado processes: set_param must reach them through a pre-step script."""
        code, _ = self.execute_vivado()
        self.assertEqual(code, 0)
        threads = 5
        self.assertEqual(self.threads_tcl.strip(), f'set_param general.maxThreads {threads}')
        hooks = {c[1]: c[2] for c in self.vivado_calls if c[0] == 'set_property' and c[1].endswith('.TCL.PRE')}
        self.assertEqual(set(hooks), {'STEPS.SYNTH_DESIGN.TCL.PRE', 'STEPS.INIT_DESIGN.TCL.PRE'})
        self.assertTrue(all(path.endswith('threads.tcl') for path in hooks.values()))
        launches = [c for c in self.vivado_calls if c[0] == 'launch_runs']
        self.assertTrue(all(c[c.index('-jobs') + 1] == '5' for c in launches))

    def test_failed_synthesis_stops_vivado(self):
        code, calls = self.execute_vivado(progress='0%')
        self.assertEqual(code, 1)
        self.assertEqual(calls.count('launch_runs'), 1)

if __name__ == '__main__':
    unittest.main()
