"""The build scripts (scripts/*.sh, fpga.sh) without AMD tools: each runs for real, in a copy
of the repository with one small kernel, against stand-ins for Vitis HLS, Vivado, Docker and
opu-system that record what they are asked and leave what the real ones leave."""
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile

from helpers import CXX, ROOT, kernel_dirs

BASH = shutil.which('bash')
PYTHON = Path(sys.executable).as_posix()

# A stand-in for vitis_hls -f hls.tcl: what the real one leaves in build/K/BOARD/.
FAKE_VITIS = r'''#!PYTHON
import os, re, sys, zipfile
from pathlib import Path
settings = Path('settings.tcl').read_text()
top = re.search(r'set cfg\(top\) (\S+)', settings).group(1)
stage = re.search(r'set cfg\(stage\) (\S+)', settings).group(1)
with open(os.environ['FAKE_LOG'], 'a') as log:
    log.write(f'vitis {" ".join(sys.argv[1:])} {stage}\n')
if stage != 'csim':
    report = Path('hls/solution/syn/report')
    report.mkdir(parents=True, exist_ok=True)
    (report / f'{top}_csynth.rpt').write_text('')
if stage == 'ip':
    ip = Path('hls/solution/impl/ip')
    ip.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ip / f'xilinx_com_hls_{top}_1_0.zip', 'w') as z:
        version = os.environ.get('FAKE_HLS_VERSION', '2023.1')
        z.writestr('component.xml', f'<xilinx:packagingInfo><xilinx:xilinxVersion>{version}</xilinx:xilinxVersion>'
                                    '</xilinx:packagingInfo>')
        src = f'drivers/{top}_v1_0/src/'
        prefix = 'X' + top[0].upper() + top[1:]
        functions = ''.join(f'void {prefix}_Set_{p}({prefix} *InstancePtr, u64 Data);\n' for p in ('din0', 'din1', 'dout'))
        z.writestr(src + f'x{top}.h', f'typedef struct {{ int x; }} {prefix};\n'
                   f'void {prefix}_Start({prefix} *InstancePtr);\n{functions}'
                   f'void {prefix}_Set_height({prefix} *InstancePtr, u32 Data);\n'
                   f'void {prefix}_Set_width({prefix} *InstancePtr, u32 Data);\n')
        z.writestr(src + f'x{top}_hw.h', f'#define X{top.upper()}_CONTROL_ADDR_AP_CTRL 0x00\n')
        z.writestr(src + f'x{top}.c', '')
        z.writestr(src + f'x{top}_linux.c', '')
        z.writestr(src + f'x{top}_sinit.c', '')
'''

# A stand-in for vivado: -source vivado.tcl (scripts/pynq.sh); -source an opu-system
# project script (scripts/opu.sh with the Vivado on PATH), which must point repoDir at opu-system.
FAKE_VIVADO = r'''#!PYTHON
import os, re, sys
from pathlib import Path
with open(os.environ['FAKE_LOG'], 'a') as log:
    log.write(f'vivado {" ".join(sys.argv[1:])}\n')
script = Path(sys.argv[sys.argv.index('-source') + 1])
if 'tcl-scripts' in script.parts:
    text = script.read_text()
    repo = re.search(r'^set repoDir (\S+)$', text, re.M).group(1)
    project = re.search(r'^set projectName (\S+)$', text, re.M).group(1)
    assert '#set repoDir /home/hypso\n' in text and Path(repo, 'vivado/run_tcl_script.sh').is_file(), text
    sdk = Path(repo, 'vivado/projects', project, project + '.sdk')
    sdk.mkdir(parents=True)
    (sdk / 'system.xsa').write_text('XSA')
    (sdk / 'bitstream.bit').write_text('BIT')
    (sdk / 'timing.rpt').write_text('All user specified timing constraints are met.')
    sys.exit(0)
impl = Path('vivado/system.runs/impl_1')
impl.mkdir(parents=True)
(impl / 'system_wrapper.bit').write_bytes(b'BIT')
handoff = Path('vivado/system.gen/sources_1/bd/system/hw_handoff')
handoff.mkdir(parents=True)
(handoff / 'system.hwh').write_text('<HWH/>')
'''

# opu-system's vivado/run_tcl_script.sh: the project's hardware files.
FAKE_RUN_TCL = r'''#!/bin/bash
set -e
here="$(cd "$(dirname "$0")" && pwd)"
script="$here/tcl-scripts/tmp/$1"
[ -f "$script" ] || { echo "no $script"; exit 1; }
echo "run_tcl_script $1" >> "$FAKE_LOG"
project=$(sed -n 's/^set projectName //p' "$script")
sdk="$here/projects/$project/$project.sdk"
mkdir -p "$sdk"
echo XSA > "$sdk/system.xsa"
echo BIT > "$sdk/bitstream.bit"
echo "${FAKE_TIMING:-All user specified timing constraints are met.}" > "$sdk/timing.rpt"
'''

# opu-system's docker/run-petalinux-docker.sh ./prepare_pl_firmware.sh XSA NAME OUT DTSI.
FAKE_PETALINUX = r'''#!/bin/bash
set -e
echo "petalinux $*" >> "$FAKE_LOG"
cd "$(dirname "$0")/../petalinux/projects"
[ "$1" = ./prepare_pl_firmware.sh ] && [ -f "$2" ] && [ -f "$5" ]
# The first run makes dtbo-project; cracklib fails without the fix (FAKE_CRACKLIB).
conf=dtbo-project/project-spec/meta-user/conf/petalinuxbsp.conf
[ -f "$conf" ] || { mkdir -p "$(dirname "$conf")"; echo '#User Configuration' > "$conf"; }
if [ -n "$FAKE_CRACKLIB" ] && ! grep -q 'branch=main' "$conf"; then exit 1; fi
echo BITBIN > "$4/$3.bit.bin"
echo "hash  /media/sd-pl/firmware/$3.bit.bin" > "$4/$3.bit.bin.md5sum"
echo DTBO > "$4/$3.dtbo"
'''

# docker: the PetaLinux image and cnn-1d's cnn-dev-image are there, opu-system's Vivado image
# with FAKE_VIVADO_IMAGE. A run of cnn-dev-image (cnn-1d's start_docker) "builds" cnn_accelerator
# in the folder it mounts, after checking the kernel's driver is there.
FAKE_DOCKER = r'''#!/bin/bash
echo "docker $*" >> "$FAKE_LOG"
if [ "$1 $2" = "image inspect" ]; then
    case "$3" in
        petalinux:2023.1|cnn-dev-image) exit 0 ;;
        vivado:2023.1) [ -n "$FAKE_VIVADO_IMAGE" ]; exit ;;
        *) exit 1 ;;
    esac
fi
if [ "$1" = run ] && [[ "$*" == *cnn-dev-image* ]]; then
    while [ $# -gt 0 ]; do [ "$1" = -v ] && mount="${2%%:*}"; shift; done
    ls "$mount"/xcnn_driver/x*.c >/dev/null && [ -f "$mount/xcnn_driver/kernel.h" ] || exit 1
    mkdir -p "$mount/build-cross" && echo ARM > "$mount/build-cross/cnn_accelerator"
fi
'''

KERNEL_CPP = 'void k_opt(const int *din0, int *dout, int height, int width) { dout[0] = din0[0] + height + width; }\n'
TESTBENCH = 'int main() { return %d; }\n'


def fake_repository(root):
    """The scripts, boards and software, one kernel and its golden one, under ROOT."""
    root = Path(root)
    for folder in ('scripts', 'boards', 'software'):
        # Not what a local build of cnn_accelerator left (git ignores it too).
        shutil.copytree(ROOT / folder, root / folder, ignore=shutil.ignore_patterns('build-cross', 'xcnn_driver'))
    (root / 'tools').mkdir()
    shutil.copy2(ROOT / 'tools/models.py', root / 'tools')
    shutil.copy2(ROOT / 'mu_sd.txt', root)
    for kernel, folder in (('k_opt', 'k_opt'), ('justoliunet_opt', 'opt'), ('justoliunet', 'golden')):
        (root / f'src/models/m/{folder}').mkdir(parents=True)
        (root / f'src/models/m/{folder}/{kernel}.cpp').write_text(KERNEL_CPP.replace('k_opt', kernel))
        (root / f'tb/{kernel}').mkdir(parents=True)
        (root / f'tb/{kernel}/{kernel}_tb.cpp').write_text(TESTBENCH % 0)
    (root / 'src/common').mkdir()
    return root


@unittest.skipIf(os.name == 'nt', 'Windows has no execute bit')
class ExecutableTests(unittest.TestCase):
    def test_every_script_is_executable(self):
        # The scripts call each other as programs, and git keeps the bit for other machines.
        # The tests below run them with bash, so they would not notice a missing one.
        scripts = sorted([*(ROOT / 'scripts').glob('*.sh'), *(ROOT / 'software/opu/scripts').glob('*.sh'),
                          *(ROOT / 'software/opu/scripts').glob('*_docker')])
        self.assertTrue(scripts)
        missing = [str(p.relative_to(ROOT)) for p in scripts if not os.access(p, os.X_OK)]
        self.assertEqual(missing, [], 'chmod +x these (and commit the mode)')


@unittest.skipIf(BASH is None, 'the scripts need bash (Git Bash, WSL or Linux)')
class ScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.root = fake_repository(self.dir / 'repo')
        self.bin = self.dir / 'bin'
        self.bin.mkdir()
        self.log = self.dir / 'log.txt'
        self.log.write_text('')
        self.fake('vitis_hls', FAKE_VITIS.replace('PYTHON', PYTHON))
        self.fake('vivado', FAKE_VIVADO.replace('PYTHON', PYTHON))
        self.fake('docker', FAKE_DOCKER)
        self.fake('python3', f'#!/bin/sh\nexec "{PYTHON}" "$@"\n')
        self.opu = self.dir / 'opu-system'
        for path, text in (('vivado/run_tcl_script.sh', FAKE_RUN_TCL),
                           ('docker/run-petalinux-docker.sh', FAKE_PETALINUX)):
            (self.opu / path).parent.mkdir(parents=True, exist_ok=True)
            (self.opu / path).write_text(text, newline='\n')
            (self.opu / path).chmod(0o755)
        (self.opu / 'petalinux/projects').mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def fake(self, name, text):
        (self.bin / name).write_text(text, newline='\n')
        (self.bin / name).chmod(0o755)

    def run_script(self, *args, env=None):
        # Only the stand-ins and the system's own tools: no real AMD tools from PATH.
        path = os.pathsep.join([self.bin.as_posix(), '/usr/bin', '/bin'])
        environment = {**os.environ, 'PATH': path, 'FAKE_LOG': self.log.as_posix(), 'FPGA_JOBS': '3',
                       'FPGA_OPU_SYSTEM': self.opu.as_posix(), **(env or {})}
        return subprocess.run([BASH, *[str(a) for a in args]], cwd=self.dir, text=True, capture_output=True,
                              env=environment)

    def script(self, name):
        return self.root / 'scripts' / name

    def out(self, kernel='k_opt', board='zcu104'):
        return self.root / 'build' / kernel / board

    # ---------------------------------------------------------------- finding things

    def test_kernels_are_the_cpp_files_in_src_models(self):
        result = subprocess.run([BASH, str(ROOT / 'scripts/kernels.sh')], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), sorted(kernel_dirs()))
        self.assertIn('boards: picobob pynqz2 zcu104 zedboard', result.stderr)

    def test_unknown_kernels_and_boards_are_named(self):
        for kernel, board, message in (('no_such', 'zcu104', 'src/models/MODEL/.../no_such.cpp'),
                                       ('../hls', 'zcu104', "no kernel '../hls'"),
                                       ('k_opt', 'zynq7030', "no board 'zynq7030': one of picobob pynqz2 zcu104 zedboard")):
            with self.subTest(kernel=kernel, board=board):
                result = self.run_script(self.script('hls.sh'), kernel, board)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_every_board_says_its_part_clock_and_flow(self):
        for board, part, flow in (('zcu104', 'xczu7ev-ffvc1156-2-e', 'pynq'), ('pynqz2', 'xc7z020clg400-1', 'pynq'),
                                  ('zedboard', 'xc7z020clg484-1', 'opu'), ('picobob', 'xc7z030sbg485-1', 'opu')):
            with self.subTest(board=board):
                result = self.run_script('-c', f'source "{self.script("common.sh")}"; load_board {board}; '
                                         'echo "$PART $CLOCK_NS $FLOW"')
                self.assertEqual(result.returncode, 0, result.stderr)
                part_found, clock, flow_found = result.stdout.split()
                self.assertEqual((part_found, flow_found), (part, flow))
                self.assertGreater(float(clock), 0)
                if flow == 'pynq':
                    self.assertTrue((ROOT / f'boards/{board}/system.tcl').is_file())

    # ---------------------------------------------------------------- test.sh

    @unittest.skipUnless(CXX, 'no C++ compiler')
    def test_test_sh_runs_the_testbench_and_fails_with_it(self):
        result = self.run_script(self.script('test.sh'), 'k_opt', env={'CXX': CXX})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASSED: k_opt', result.stdout)
        self.assertTrue((self.root / 'build/k_opt/native/testbench.log').is_file())
        (self.root / 'tb/k_opt/k_opt_tb.cpp').write_text(TESTBENCH % 3)
        result = self.run_script(self.script('test.sh'), 'k_opt', env={'CXX': CXX})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('the testbench failed', result.stderr)

    # ---------------------------------------------------------------- hls.sh

    def test_hls_writes_the_settings_and_leaves_the_export(self):
        result = self.run_script(self.script('hls.sh'), 'k_opt', 'pynqz2')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        out = self.out(board='pynqz2')
        settings = (out / 'settings.tcl').read_text()
        for line in ('set cfg(stage) ip', 'set cfg(top) k_opt', 'set cfg(part) xc7z020clg400-1',
                     'set cfg(clock_ns) 12.8', 'set cfg(cosim) 0', f'set cfg(run_dir) {{{out.as_posix()}}}',
                     f'set cfg(sources) [list {{{(self.root / "src/models/m/k_opt/k_opt.cpp").as_posix()}}}]'):
            self.assertIn(line, settings)
        self.assertTrue((out / 'hls.tcl').is_file())
        self.assertTrue(zipfile.is_zipfile(out / 'export.zip'))
        self.assertIn('clock_ns: 12.8', (out / 'build.txt').read_text())
        self.assertIn('Next: scripts/pynq.sh k_opt pynqz2', result.stdout)

    def test_hls_stage_and_options(self):
        for args, want in ((['csim'], ['set cfg(stage) csim', 'set cfg(cosim) 0']),
                           (['cosim', '--dataflow-profile'], ['set cfg(stage) cosim', 'set cfg(cosim) 1',
                                                              'set cfg(profile) 1']),
                           (['--cosim', '--clock-ns', '7.5'], ['set cfg(stage) ip', 'set cfg(cosim) 1',
                                                               'set cfg(clock_ns) 7.5'])):
            with self.subTest(args=args):
                result = self.run_script(self.script('hls.sh'), 'k_opt', 'zcu104', *args)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                settings = (self.out() / 'settings.tcl').read_text()
                for line in want:
                    self.assertIn(line, settings)
                self.assertEqual((self.out() / 'export.zip').exists(), 'csim' not in args and 'cosim' not in args)
        result = self.run_script(self.script('hls.sh'), 'k_opt', 'zcu104', 'bitstream')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('usage', result.stderr)

    # ---------------------------------------------------------------- pynq.sh

    def pynq_package(self, kernel):
        result = self.run_script(self.script('build.sh'), kernel, 'zcu104')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with zipfile.ZipFile(self.out(kernel) / f'{kernel}_pynq.zip') as z:
            return sorted(n for n in z.namelist() if not n.endswith('/')), result

    def test_pynq_package_of_an_optimized_kernel(self):
        names, result = self.pynq_package('justoliunet_opt')
        k = 'justoliunet_opt'
        self.assertEqual(names, sorted(f'{k}_pynq/{n}' for n in (f'{k}.bit', f'{k}.hwh', f'x{k}_hw.h',
                                                                  'classify.ipynb', 'kernel.json')))
        package = self.out(k) / f'{k}_pynq'
        self.assertEqual(json.loads((package / 'kernel.json').read_text()),
                         {'kernel': k, 'model': 'justoliunet', 'patch': None, 'out_frac': 16})
        self.assertEqual((package / f'{k}.bit').read_bytes(), b'BIT')
        settings = (self.out(k) / 'vivado-settings.tcl').read_text()
        for line in ('set cfg(jobs) 3', 'set cfg(clock_ns) 5.0', f'set cfg(ip_repo) {{{(self.out(k) / "ip").as_posix()}}}',
                     f'set cfg(board_script) {{{(self.root / "boards/zcu104/system.tcl").as_posix()}}}'):
            self.assertIn(line, settings)
        self.assertTrue((self.out(k) / 'ip/component.xml').is_file(), 'the export, unpacked for Vivado')
        self.assertIn(f'then open {k}_pynq/classify.ipynb', result.stdout)

    def test_pynq_package_of_a_golden_kernel_and_of_no_model(self):
        names, _ = self.pynq_package('justoliunet')
        self.assertIn('justoliunet_pynq/classify_golden.ipynb', names)
        self.assertIn('justoliunet_pynq/mu_sd.txt', names)
        names, _ = self.pynq_package('k_opt')
        self.assertEqual(names, sorted(f'k_opt_pynq/{n}' for n in ('k_opt.bit', 'k_opt.hwh', 'xk_opt_hw.h')))

    def test_pynq_package_again_without_vivado(self):
        self.pynq_package('justoliunet_opt')
        self.log.write_text('')
        (self.root / 'software/pynq/classify.ipynb').write_text('{"edited": 1}')
        result = self.run_script(self.script('pynq.sh'), 'justoliunet_opt', 'zcu104', '--package')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('vivado', self.log.read_text())
        self.assertEqual((self.out('justoliunet_opt') / 'justoliunet_opt_pynq/classify.ipynb').read_text(),
                         '{"edited": 1}')

    def test_pynq_needs_an_export_and_a_pynq_board(self):
        result = self.run_script(self.script('pynq.sh'), 'k_opt', 'zcu104')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('export.zip', result.stderr)
        result = self.run_script(self.script('pynq.sh'), 'k_opt', 'picobob')
        self.assertIn('scripts/opu.sh k_opt picobob', result.stderr)

    # ---------------------------------------------------------------- opu.sh

    def opu_build(self, board='picobob', env=None, kernel='k_opt'):
        # opu-system's own Vivado 2023.1 is there unless a test says otherwise.
        return self.run_script(self.script('build.sh'), kernel, board, env={'FAKE_VIVADO_IMAGE': '1', **(env or {})})

    def test_opu_builds_through_opu_system_and_packs_the_tarball(self):
        result = self.opu_build()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        project = 'k_opt_picobob'
        # 1. The project script, as opu-system's own, in its tcl-scripts/tmp/.
        tcl = (self.opu / f'vivado/tcl-scripts/tmp/{project}.tcl').read_text()
        for line in (f'set projectName {project}', 'set hlsTop k_opt', 'set hlsClockNs 10.0',
                     'source $repoDir/vivado/tcl-scripts/picozed_project.tcl',
                     'source $repoDir/vivado/tcl-scripts/opu_base.tcl',
                     'source $repoDir/vivado/tcl-scripts/tmp/add_hls_kernel.tcl',
                     'source $repoDir/vivado/tcl-scripts/finish_design.tcl',
                     # run_tcl_script.sh's sed turns these two into the path it runs from
                     '#set repoDir /home/hypso/opu-system', 'set repoDir /home/hypso'):
            self.assertIn(line, tcl)
        self.assertIsNone(re.search('@[A-Z_]+@', tcl))
        self.assertTrue((self.opu / 'vivado/tcl-scripts/tmp/add_hls_kernel.tcl').is_file())
        self.assertTrue((self.opu / f'vivado/projects/{project}_hls_export/component.xml').is_file())
        # 2. PetaLinux, with the xsa and the overlay's additions at the board's reserved memory.
        pl = self.opu / f'petalinux/projects/{project}'
        dtsi = (pl / f'{project}.dtsi').read_text()
        self.assertIn('reg = <0x20000000 0x09600000 0x29600000 0x01000000>;', dtsi)
        self.assertIsNone(re.search('@[A-Z_]+@', dtsi))
        self.assertIn(f'petalinux ./prepare_pl_firmware.sh ./{project}/{project}.xsa k_opt ./{project}/output '
                      f'./{project}/{project}.dtsi', self.log.read_text())
        # 3. cnn_accelerator, built by cnn-1d's scripts with the kernel's driver and kernel.h.
        driver = self.out(board='picobob') / 'driver'
        self.assertEqual(sorted(p.name for p in driver.iterdir()),
                         ['kernel.h', 'xk_opt.c', 'xk_opt.h', 'xk_opt_hw.h', 'xk_opt_linux.c', 'xk_opt_sinit.c'])
        self.assertIn('#define KERNEL_CLOCK_MHZ 100.000', (driver / 'kernel.h').read_text())
        self.assertIn('cnn-dev-image /bin/bash -c cd build-cross && cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-arm.cmake',
                      self.log.read_text())
        # 4. The tarball, to unpack in /media/sd-pl.
        with tarfile.open(self.out(board='picobob') / 'k_opt_opu.tar.gz') as tar:
            names = sorted(m.name for m in tar.getmembers() if m.isfile())
        self.assertEqual(names, ['firmware/k_opt.bit.bin', 'firmware/k_opt.bit.bin.md5sum', 'firmware/k_opt.dtbo',
                                 'k_opt/cnn_accelerator', 'k_opt/load.sh', 'k_opt/process-capture.sh'])
        self.assertIn('obip fpga reload k_opt.bit.bin -o k_opt.dtbo', result.stdout)

    def test_opu_zedboard_project_and_memory(self):
        result = self.opu_build('zedboard')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        tcl = (self.opu / 'vivado/tcl-scripts/tmp/k_opt_zedboard.tcl').read_text()
        self.assertIn('source $repoDir/vivado/tcl-scripts/zedboard_project.tcl', tcl)
        self.assertIn('set hlsClockNs 12.8', tcl)
        dtsi = (self.opu / 'petalinux/projects/k_opt_zedboard/k_opt_zedboard.dtsi').read_text()
        self.assertIn('reg = <0x10000000 0x09600000 0x19600000 0x01000000>;', dtsi)
        self.assertIn('#define KERNEL_CLOCK_MHZ 78.125', (self.out(board='zedboard') / 'driver/kernel.h').read_text())

    def test_opu_uses_opu_systems_vivado_for_any_export(self):
        # The FPGA workstation's way: an export from Vitis HLS 2025.2, opu-system's Vivado 2023.1.
        result = self.opu_build(env={'FAKE_HLS_VERSION': '2025.2'})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('run_tcl_script k_opt_picobob.tcl', self.log.read_text())
        self.assertNotIn('-mode batch -source', self.log.read_text())

    def test_opu_without_opu_systems_vivado_runs_the_one_on_path(self):
        result = self.opu_build(env={'FAKE_VIVADO_IMAGE': ''})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        log = self.log.read_text()
        self.assertNotIn('run_tcl_script', log)
        self.assertIn('vivado -notrace -nolog -nojou -quiet -mode batch -source', log)
        self.assertIn('petalinux ./prepare_pl_firmware.sh', log)
        self.assertEqual(list((self.opu / 'vivado/tcl-scripts/tmp').glob('local-*')), [])

    def test_opu_fixes_cracklib_and_runs_petalinux_again(self):
        result = self.opu_build(env={'FAKE_CRACKLIB': '1'})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.log.read_text().count('petalinux ./prepare_pl_firmware.sh'), 2)
        conf = (self.opu / 'petalinux/projects/dtbo-project/project-spec/meta-user/conf/petalinuxbsp.conf').read_text()
        for recipe in ('cracklib', 'cracklib-native'):
            self.assertIn(f'SRC_URI:remove:pn-{recipe} = "git://github.com/cracklib/cracklib;protocol=https;branch=master"', conf)
            self.assertIn(f'SRC_URI:prepend:pn-{recipe} = "git://github.com/cracklib/cracklib;protocol=https;branch=main "', conf)
        # The next build has the fix from the start, and adds it only once.
        self.log.write_text('')
        result = self.opu_build(env={'FAKE_CRACKLIB': '1'})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.log.read_text().count('petalinux ./prepare_pl_firmware.sh'), 1)
        conf = (self.opu / 'petalinux/projects/dtbo-project/project-spec/meta-user/conf/petalinuxbsp.conf').read_text()
        self.assertEqual(conf.count('branch=main'), 2)

    def test_opu_stops_when_timing_fails(self):
        result = self.opu_build(env={'FAKE_TIMING': 'Timing constraints are not met.'})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('fails timing', result.stderr)
        self.assertNotIn('petalinux', self.log.read_text())

    def test_opu_driver_again_without_vivado(self):
        self.assertEqual(self.opu_build().returncode, 0)
        self.log.write_text('')
        result = self.run_script(self.script('opu.sh'), 'k_opt', 'picobob', '--driver')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('run_tcl_script', self.log.read_text())
        self.assertNotIn('petalinux', self.log.read_text())
        self.assertTrue((self.out(board='picobob') / 'k_opt_opu.tar.gz').is_file())

    def test_opu_needs_opu_system_and_an_opu_board(self):
        result = self.run_script(self.script('hls.sh'), 'k_opt', 'picobob')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = self.run_script(self.script('opu.sh'), 'k_opt', 'picobob', env={'FPGA_OPU_SYSTEM': '/nowhere'})
        self.assertIn('no opu-system in /nowhere', result.stderr)
        result = self.run_script(self.script('opu.sh'), 'k_opt', 'zcu104')
        self.assertIn('scripts/pynq.sh k_opt zcu104', result.stderr)
        result = self.run_script(self.script('opu.sh'), 'k_opt', 'zedboard', '--driver')
        self.assertIn('export.zip', result.stderr)

    # ------------------------------------------------------ get-petalinux.sh

    def test_get_petalinux_copies_the_image_over_ssh(self):
        # ssh runs the command here; docker keeps "the image" in a file once loaded.
        self.fake('ssh', '#!/bin/bash\necho "ssh $1" >> "$FAKE_LOG"\nshift\nREMOTE=1 eval "$*"\n')
        self.fake('zstd', '#!/bin/bash\ncat\n')
        self.fake('getenforce', '#!/bin/sh\necho Permissive\n')
        self.fake('docker', r'''#!/bin/bash
image="$FAKE_LOG.image"
case "$1 $2" in
    "image inspect") { [ -n "$REMOTE" ] || [ -f "$image" ]; } && echo "sha256:a sha256:b" ;;
    "save "*) echo IMAGE ;;
    "load ") cat > "$image"; echo "docker load" >> "$FAKE_LOG" ;;
esac
''')
        result = self.run_script(self.script('get-petalinux.sh'), 'lab')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('copied, the same as on lab', result.stdout)
        self.assertEqual(Path(self.log.as_posix() + '.image').read_text(), 'IMAGE\n')
        self.assertIn('ssh lab', self.log.read_text())
        # Once here, nothing is copied again.
        self.log.write_text('')
        result = self.run_script(self.script('get-petalinux.sh'))
        self.assertIn('already here', result.stdout)
        self.assertNotIn('ssh', self.log.read_text())


@unittest.skipIf(BASH is None, 'fpga.sh needs bash (Git Bash, WSL or Linux)')
class FpgaShellTests(unittest.TestCase):
    def bash(self, script, cwd=None):
        prefix = (f'export FPGA_ROOT="{ROOT.as_posix()}"\n'
                  f'export FPGA_PYTHON="{PYTHON}"\n'  # python3 may be a stub on Windows
                  f'source "{(ROOT / "fpga.sh").as_posix()}"\n')
        return subprocess.run([BASH, '-c', prefix + script], text=True, capture_output=True, cwd=cwd)

    def complete(self, *words):
        """What Tab offers after `words` (the last one is the word being completed)."""
        array = ' '.join(f'"{w}"' for w in words)
        result = self.bash(f'COMP_WORDS=({array}); COMP_CWORD={len(words) - 1}; _fpga_complete; '
                           'printf "%s\\n" "${COMPREPLY[@]}"')
        self.assertEqual(result.returncode, 0, result.stderr)
        return sorted(result.stdout.split())

    def test_help_lists_every_command(self):
        result = self.bash('fpga_help')
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ('fpga_kernels', 'fpga_env', 'fpga_test', 'fpga_build', 'fpga_ip', 'fpga_cosim',
                        'fpga_bitstream', 'fpga_package', 'fpga_zip'):
            self.assertIn(command, result.stdout)

    def test_tab_completes_kernels_then_boards_then_options(self):
        for command in ('fpga_test', 'fpga_build', 'fpga_ip', 'fpga_cosim', 'fpga_bitstream', 'fpga_package'):
            with self.subTest(command=command):
                self.assertEqual(self.complete(command, 'justounetsimple'),
                                 ['justounetsimple', 'justounetsimple_opt', 'justounetsimple_opt_tiled',
                                  'justounetsimple_opt_zynq7020', 'justounetsimple_opt_zynq7030'])
        self.assertEqual(self.complete('fpga_build', 'justounetsimple', ''), ['picobob', 'pynqz2', 'zcu104', 'zedboard'])
        self.assertEqual(self.complete('fpga_ip', 'justounetsimple', 'z'), ['zcu104', 'zedboard'])
        self.assertEqual(self.complete('fpga_build', 'justounetsimple', 'picobob', '--'), ['--clock-ns', '--cosim'])
        self.assertEqual(self.complete('fpga_cosim', 'justounetsimple', 'zcu104', '--d'), ['--dataflow-profile'])

    def test_the_board_defaults_to_the_zcu104(self):
        result = self.bash('_fpga_board_args k; _fpga_board_args k --clock-ns 4; _fpga_board_args k picobob --cosim')
        self.assertEqual(result.stdout.splitlines(), ['k zcu104', 'k zcu104 --clock-ns 4', 'k picobob --cosim'])
        result = self.bash('_fpga_flow picobob; _fpga_flow zcu104')
        self.assertEqual(result.stdout.split(), ['opu', 'pynq'])

    def test_a_kernel_must_be_named(self):
        for command in ('fpga_test', 'fpga_build', 'fpga_ip', 'fpga_cosim', 'fpga_bitstream', 'fpga_package'):
            with self.subTest(command=command):
                result = self.bash(command)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('usage', result.stderr)

    def test_completion_is_registered(self):
        result = self.bash('complete -p fpga_build fpga_zip')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('complete -F _fpga_complete fpga_build', result.stdout)
        self.assertIn('complete -d fpga_zip', result.stdout)

    def test_zip_keeps_the_folder_name_inside(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'aegean').mkdir()
            (Path(tmp) / 'aegean/cube.npy').write_bytes(b'')
            result = self.bash('fpga_zip aegean', cwd=tmp)
            self.assertEqual(result.returncode, 0, result.stderr)
            with zipfile.ZipFile(Path(tmp) / 'aegean.zip') as zip_file:
                self.assertIn('aegean/cube.npy', zip_file.namelist())


if __name__ == '__main__':
    unittest.main()
