#!/usr/bin/env python3
"""Build an HLS kernel: from C++ to a ZCU104 bitstream and a zip for the PYNQ notebook.

    python3 main.py STAGE --kernel KERNEL

Each stage includes the ones above it:

    native     compile src/hls/KERNEL/ + tb/KERNEL/ with g++ and run the testbench
    csim       the same testbench, inside Vitis HLS                     (scripts/hls.tcl)
    csynth     ... then C++ -> Verilog
    cosim      ... then simulate that Verilog with the testbench
    ip         ... then export the IP (hls/solution/impl/ip/), for a block design of your
               own; stops before Vivado
    bitstream  ... then the block design around that IP, synthesis, place and
               route, .bit (scripts/vivado.tcl + boards/BOARD/system.tcl); the PYNQ zip
               (for a board running the OPU system's PetaLinux, the OPU zip)

and two helpers:

    pynq       re-make the PYNQ (or OPU) zip of an earlier bitstream run (--run, default the newest)
    kernels    list the kernels and the files found for each

--board picks the FPGA (its part and board script, from config/project.json); the
default is the config's "board". Every run gets its own folder, build/KERNEL/DATE-STAGE/
(DATE-STAGE-BOARD for a board other than the default), with everything it produced in it.
build/KERNEL/latest points at the newest one. Settings are in config/project.json.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
import zipfile

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'config' / 'project.json').read_text(encoding='utf-8'))
STAGES = ('native', 'csim', 'csynth', 'cosim', 'ip', 'bitstream')


# ------------------------------------------------------------------ 1. The kernel's files

def find_kernel(name):
    """The C++ for a kernel, found by folder name: src/hls/NAME/ and tb/NAME/."""
    source_dir, testbench_dir = ROOT / 'src' / 'hls' / name, ROOT / 'tb' / name
    if not re.fullmatch(r'\w+', name) or not source_dir.is_dir():
        raise SystemExit(f'No kernel {name!r}: expected a folder src/hls/{name}/ ("main.py kernels" lists them)')
    directives = ROOT / 'config' / f'{name}.tcl'
    kernel = {
        'name': name,
        # The top function is named after the folder, unless config/project.json says otherwise.
        'top': CONFIG['kernels'].get(name, {}).get('top', name),
        'sources': sorted(source_dir.glob('*.cpp')),
        'testbench': sorted(testbench_dir.glob('*.cpp')),
        'include_dirs': [source_dir, ROOT / 'src' / 'common'],
        # Optional HLS directives (set_directive_* commands) instead of pragmas.
        'directives': directives if directives.is_file() else None,
    }
    if not kernel['sources']:
        raise SystemExit(f'No .cpp files in src/hls/{name}/')
    if not kernel['testbench']:
        raise SystemExit(f'No testbench: add a .cpp with a main() to tb/{name}/ that returns nonzero on failure')
    return kernel


# ------------------------------------------------------------------ 2. A folder for this run

def new_run_folder(name, stage, board=None):
    """build/NAME/2026-10-06_14-03-12-STAGE/ (…-STAGE-BOARD for another board than the
    default), and build/NAME/latest pointing at it."""
    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if board and board != CONFIG['board']:
        stage = f'{stage}-{board}'
    run = ROOT / 'build' / name / f'{stamp}-{stage}'
    count = 2
    while run.exists():  # two runs in the same second
        run = run.with_name(f'{stamp}-{stage}-{count}')
        count += 1
    run.mkdir(parents=True)
    latest = run.parent / 'latest'
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run.name, target_is_directory=True)
    except OSError:  # Windows without developer mode cannot make symlinks; not worth failing over
        pass
    return run


# ------------------------------------------------------------------ 3. Running a tool

def show(path):
    """A path as printed: relative to the repository when it is inside it."""
    path = Path(path)
    return (path.relative_to(ROOT) if path.is_relative_to(ROOT) else path).as_posix()


def run_tool(command, run, log_name, env=None):
    """Run a command in the run folder, showing its output and saving it to LOG_NAME."""
    command = [str(part) for part in command]
    print('> ' + ' '.join(command), flush=True)
    with open(run / log_name, 'w', encoding='utf-8') as log, \
            subprocess.Popen(command, cwd=run, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding='utf-8', errors='replace') as process:
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
    if process.returncode:
        raise SystemExit(f'{Path(command[0]).name} failed (exit code {process.returncode}); '
                         f'see {show(run / log_name)}')


def find_tool(name):
    path = shutil.which(name)
    if not path:
        raise SystemExit(f'{name} not found on PATH. For the AMD tools, source settings64.sh first '
                         '(fpga_env); for g++, install it or set CXX to another compiler.')
    return path


# ------------------------------------------------------------------ 4a. native: g++ only

def native(kernel, run):
    compiler = find_tool(os.environ.get('CXX', 'g++'))
    # On Windows the testbench needs the compiler's own runtime DLLs, so put its folder first.
    env = dict(os.environ, PATH=str(Path(compiler).parent) + os.pathsep + os.environ.get('PATH', ''))
    program = run / ('testbench.exe' if os.name == 'nt' else 'testbench')
    run_tool([compiler, '-std=c++14', '-O2', '-Wall', '-Wextra', '-Wno-unknown-pragmas', '-Wno-unused-label',
              *[f'-I{d}' for d in kernel['include_dirs']], *kernel['sources'], *kernel['testbench'],
              '-o', program], run, 'compile.log', env)
    run_tool([program], run, 'testbench.log', env)


# ------------------------------------------------------------------ 4b. Vitis HLS and Vivado

def job_count(setting=None):
    """config "jobs": a number, or "nproc": every core this process may use, as `nproc` counts
    them (the CPUs it is allowed on, not just the CPUs the machine has)."""
    setting = CONFIG['jobs'] if setting is None else setting
    if setting == 'nproc':
        return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    return int(setting)


def board_clock_ns(board):
    """The default clock period for BOARD: its own "clock_ns" in the config, else the global one
    (a 7-series Zynq does not reach what the ZCU104 does)."""
    return float(CONFIG['boards'][board].get('clock_ns', CONFIG['clock_ns']))


def write_settings(kernel, run, stage, clock_ns, skip_cosim, board=None, profile=False):
    """settings.tcl: everything the Tcl scripts need to know, as the Tcl array cfg(...)."""
    target = CONFIG['boards'][board or CONFIG['board']]
    def tcl(value):  # a {braced} Tcl word: taken literally, spaces and all
        value = value.as_posix() if isinstance(value, Path) else str(value)
        if any(c in value for c in '{}\\'):
            raise SystemExit(f'Path or setting with a brace or backslash cannot go to Tcl: {value}')
        return '{' + value + '}'

    settings = {
        'root': ROOT, 'run_dir': run, 'stage': stage, 'top': kernel['top'],
        'board': board or CONFIG['board'], 'part': target['part'], 'clock_ns': clock_ns, 'jobs': job_count(),
        'skip_cosim': int(skip_cosim), 'profile': int(profile), 'board_script': ROOT / target['script'],
        'directives': kernel['directives'] or '',
    }
    lines = [f'set cfg({key}) {tcl(value)}' for key, value in settings.items()]
    for key in ('sources', 'testbench', 'include_dirs'):
        lines.append(f'set cfg({key}) [list {" ".join(tcl(path) for path in kernel[key])}]')
    (run / 'settings.tcl').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def hls(run):
    # 2024.x calls Vitis HLS through vitis-run; 2023.x and older only have vitis_hls.
    if shutil.which('vitis-run'):
        run_tool([find_tool('vitis-run'), '--mode', 'hls', '--tcl', 'hls.tcl'], run, 'hls.log')
    else:
        run_tool([find_tool('vitis_hls'), '-f', 'hls.tcl'], run, 'hls.log')


def vivado(run):
    run_tool([find_tool('vivado'), '-mode', 'batch', '-source', 'vivado.tcl', '-notrace',
              '-log', 'vivado.log', '-journal', 'vivado.jou'], run, 'vivado-console.log')


def print_estimates(run, top):
    """The numbers from HLS synthesis worth seeing right away; the full report is next to it."""
    report = run / 'hls' / 'solution' / 'syn' / 'report' / f'{top}_csynth.xml'
    try:
        root = ElementTree.parse(report).getroot()
    except (OSError, ElementTree.ParseError):  # no synthesis in this run, or it failed
        return
    latency = root.find('PerformanceEstimates/SummaryOfOverallLatency')
    clock = root.findtext('PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod')
    used, available = root.find('AreaEstimates/Resources'), root.find('AreaEstimates/AvailableResources')
    print(f'HLS estimates ({show(report.with_suffix(".rpt"))}):')
    if latency is not None:
        print(f'  latency   {latency.findtext("Worst-caseLatency")} cycles, '
              f'a new start every {latency.findtext("Interval-max")} cycles')
    if clock:
        print(f'  clock     {clock} ns needed, {root.findtext("UserAssignments/TargetClockPeriod")} ns target')
    if used is not None and available is not None:
        print('  resources ' + '  '.join(f'{r.tag} {r.text}/{available.findtext(r.tag)}' for r in used))


# ------------------------------------------------------------------ 5. The zip for the board

def software_files(kind, name):
    """What software/KIND/NAME/ holds for the board, and the other repository files its
    include.txt names (one per line, # comments)."""
    folder = ROOT / 'software' / kind / name
    files = [f for f in sorted(folder.glob('*')) if f.is_file() and f.name != 'include.txt'] if folder.is_dir() else []
    include = folder / 'include.txt'
    if include.is_file():
        for line in include.read_text(encoding='utf-8').splitlines():
            line = line.split('#')[0].strip()
            if line:
                files.append(ROOT / line)
    return files


def package_pynq(run, name):
    """RUN/NAME_pynq/ and RUN/NAME_pynq.zip: everything the notebook on the board needs."""
    def one(pattern, what):
        found = sorted(run.glob(pattern))
        if not found:
            raise SystemExit(f'No {what} in {show(run)}/{pattern}')
        return found[0]

    bit = one('vivado/system.runs/impl_1/*.bit', 'bitstream')
    hwh = one('vivado/**/hw_handoff/system.hwh', 'block design description')
    header = one('hls/solution/impl/ip/drivers/*/src/*_hw.h', 'register map')

    out = run / f'{name}_pynq'
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    shutil.copy2(bit, out / f'{name}.bit')  # PYNQ finds the .hwh by the .bit's name
    shutil.copy2(hwh, out / f'{name}.hwh')
    shutil.copy2(header, out / header.name)
    for file in software_files('pynq', name):
        shutil.copy2(file, out / file.name)

    archive = run / f'{name}_pynq.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in sorted(out.iterdir()):
            zip_file.write(file, f'{name}_pynq/{file.name}')
    print(f'\nFor the board: {show(archive)}')
    print(f'  Upload it in Jupyter (http://BOARD:9090), run  !unzip -o {name}_pynq.zip  in a cell,')
    for notebook_file in sorted(out.glob('*.ipynb')):   # its own, or one include.txt names
        print(f'  then open {name}_pynq/{notebook_file.name} and run it from the top.')
    return archive


# The OPU system (HYPSO's PetaLinux, on the lab's ZedBoards) loads designs at run time
# with fpgautil, from here (opu-system's fpga_reconfig.md).
OPU_FIRMWARE = '/media/sd-pl/firmware'


def bit_to_bin(bit):
    """A .bit file's configuration data as the .bit.bin fpgautil loads (what bootgen
    -process_bitstream bin makes): without the header, each 32-bit word byte-swapped, so
    the sync word AA 99 55 66 reads 66 55 99 AA, as the Zynq FPGA manager looks for it."""
    i = 13                                   # field 1 (9 bytes) and its length, field 2's length
    while i < len(bit):
        key = bit[i:i + 1]
        if key == b'e':                      # the data: a 4-byte length, then the words
            n = int.from_bytes(bit[i + 1:i + 5], 'big')
            data = bit[i + 5:i + 5 + n]
            if n % 4 or len(data) != n:
                break
            return b''.join(data[k:k + 4][::-1] for k in range(0, n, 4))
        if key not in (b'a', b'b', b'c', b'd'):   # design, part, date, time: 2-byte lengths
            break
        i += 3 + int.from_bytes(bit[i + 1:i + 3], 'big')
    raise SystemExit('Not a bitstream (.bit) file: no configuration data found')


def hwh_kernel(hwh):
    """From the block design's .hwh: (base, size) of the kernel's registers as the PS sees
    them, and FCLK_CLK0 in Hz as the PS was set to make it."""
    root = ElementTree.parse(hwh).getroot()
    ranges = [m for m in root.iter('MEMRANGE')
              if m.get('INSTANCE') == 'kernel' and m.get('MEMTYPE') == 'REGISTER']
    rates = [p.get('VALUE') for p in root.iter('PARAMETER') if p.get('NAME') == 'PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ']
    if not ranges or not rates:
        raise SystemExit(f'{show(hwh)}: no kernel registers or no FCLK_CLK0 rate in it')
    base, high = int(ranges[0].get('BASEVALUE'), 16), int(ranges[0].get('HIGHVALUE'), 16)
    # Rounded down: Linux picks the nearest rate at or below it, never faster than HLS planned.
    return base, high - base + 1, int(float(rates[0]) * 1e6)


def opu_registers(header):
    """kernel_regs.h for the board's C programs, from the text of x<kernel>_hw.h: the
    offsets classify.c uses, under names that do not depend on the kernel."""
    addr = dict(re.findall(r'_ADDR_(\w+)\s+(0x[0-9a-fA-F]+)', header))
    bits = dict(re.findall(r'_BITS_(\w+)\s+(\d+)', header))
    dins = sorted((k for k in addr if re.fullmatch(r'DIN\d+_DATA', k)), key=lambda k: int(k[3:-5]))
    missing = [k for k in ('AP_CTRL', 'DOUT_DATA', 'HEIGHT_DATA', 'WIDTH_DATA') if k not in addr]
    if missing or not dins:
        raise SystemExit(f'The register map has no {", ".join(missing or ["DIN0_DATA"])}')
    return ('/* The kernel\'s registers, from its x..._hw.h, for the board\'s programs. Made by main.py. */\n'
            f'#define REG_AP_CTRL {addr["AP_CTRL"]}\n'
            f'#define REG_HEIGHT {addr["HEIGHT_DATA"]}\n'
            f'#define REG_WIDTH {addr["WIDTH_DATA"]}\n'
            f'#define REG_DINS {{{", ".join(addr[k] for k in dins)}}}   /* every input port */\n'
            f'#define REG_DIN_64 {int(bits.get(dins[0]) == "64")}   /* addresses in two registers */\n'
            f'#define REG_DOUT {addr["DOUT_DATA"]}\n'
            f'#define REG_DOUT_64 {int(bits.get("DOUT_DATA") == "64")}\n')


def opu_compiler():
    """A C compiler for the board's ARM (Cortex-A9, hard float, Linux): $OPU_CC, else one on PATH."""
    if os.environ.get('OPU_CC'):
        return os.environ['OPU_CC']
    for name in ('arm-linux-gnueabihf-gcc', 'arm-none-linux-gnueabihf-gcc'):
        if shutil.which(name):
            return shutil.which(name)
    return None


def package_opu(run, name):
    """RUN/NAME_opu/ and RUN/NAME_opu.zip: the design for the OPU system's PetaLinux, to
    copy to OPU_FIRMWARE on the board and load with its load.sh."""
    def one(pattern, what):
        found = sorted(run.glob(pattern))
        if not found:
            raise SystemExit(f'No {what} in {show(run)}/{pattern}')
        return found[0]

    bit = one('vivado/system.runs/impl_1/*.bit', 'bitstream')
    hwh = one('vivado/**/hw_handoff/system.hwh', 'block design description')
    header = one('hls/solution/impl/ip/drivers/*/src/*_hw.h', 'register map')
    base, size, rate = hwh_kernel(hwh)

    out = run / f'{name}_opu'
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    firmware = f'{name}.bit.bin'
    data = bit_to_bin(bit.read_bytes())
    (out / firmware).write_bytes(data)
    # obip fpga reload and load.sh check the file against this before loading it.
    md5 = hashlib.md5(data).hexdigest()
    (out / f'{firmware}.md5sum').write_text(f'{md5}  {OPU_FIRMWARE}/{firmware}\n', encoding='utf-8')
    shutil.copy2(header, out / header.name)

    # The kernel's programs (software/opu/NAME/), built for the board: it has no compiler.
    # Static, so they do not depend on the board's C library.
    programs = [f for f in software_files('opu', name) if f.suffix == '.c']
    compiler = opu_compiler() if programs else None
    if programs:
        (out / 'kernel_regs.h').write_text(opu_registers(header.read_text(encoding='utf-8')), encoding='utf-8')
    for source in programs:
        shutil.copy2(source, out / source.name)
        if compiler:
            subprocess.run([compiler, '-O2', '-Wall', '-static', f'-I{out}', str(source), '-o',
                            str(out / source.stem)], check=True)

    # The overlay: which bitstream the full region gets, FCLK_CLK0 (set by the boot loader for
    # the design the board booted with, so set here; the xlnx,fclk driver applies it) and the
    # kernel's registers, for UIO (generic-uio, as opu-system does for CubeDMA).
    dtsi = out / f'{name}.dtsi'
    dtsi.write_text(f'''/* {name} on the OPU system's PetaLinux. Made by main.py from {hwh.name}.
 * Load: fpgautil -b {firmware} -o {name}.dtbo (load.sh does it). */
/dts-v1/;
/plugin/;
/ {{
	fragment@0 {{
		target = <&fpga_full>;
		__overlay__ {{
			firmware-name = "{firmware}";
		}};
	}};
	fragment@1 {{
		target = <&amba>;
		__overlay__ {{
			#address-cells = <1>;
			#size-cells = <1>;
			clocking0 {{
				compatible = "xlnx,fclk";
				#clock-cells = <0>;
				clocks = <&clkc 15>;
				clock-output-names = "fabric_clk";
				assigned-clocks = <&clkc 15>;
				assigned-clock-rates = <{rate}>;
			}};
			kernel@{base:x} {{
				compatible = "generic-uio";
				reg = <0x{base:x} 0x{size:x}>;
			}};
		}};
	}};
}};
''', encoding='utf-8', newline='\n')
    dtc = shutil.which('dtc')
    if dtc:
        subprocess.run([dtc, '-@', '-q', '-I', 'dts', '-O', 'dtb', '-o', str(out / f'{name}.dtbo'), str(dtsi)],
                       check=True)

    (out / 'load.sh').write_text(f'''#!/bin/sh
# Load {name} on the FPGA: run as root on the board, with this folder's files in
# {OPU_FIRMWARE}. Made by main.py.
set -e
cd {OPU_FIRMWARE}
[ -f {name}.dtbo ] || dtc -@ -q -I dts -O dtb -o {name}.dtbo {name}.dtsi
md5sum -c {firmware}.md5sum
# An overlay already in the full region (the OPU base design's, loaded at boot) blocks a
# new one; it goes, and with it the drivers of that design's UART and GPIOs.
for overlay in /configfs/device-tree/overlays/* /sys/kernel/config/device-tree/overlays/*; do
    if [ -d "$overlay" ]; then echo "removing the overlay $overlay"; rmdir "$overlay"; fi
done
fpgautil -b {firmware} -o {name}.dtbo
echo "FPGA: $(cat /sys/class/fpga_manager/fpga0/state)"
# FCLK_CLK0 in kHz: the IO PLL (33.333 MHz PS_CLK x FDIV) through its two dividers.
fclk_khz() {{
    io=$(devmem 0xF8000108)
    fclk=$(devmem 0xF8000170)
    echo $((33333 * ((io >> 12) & 0x7f) / ((fclk >> 8) & 0x3f) / ((fclk >> 20) & 0x3f)))
}}
if [ "$(fclk_khz)" -gt {rate // 1000} ]; then
    # The overlay's rate did not take (no xlnx,fclk driver): the IO PLL's smallest divider
    # that is not faster than the kernel was built for, written to the SLCR (unlocked first).
    io=$(devmem 0xF8000108)
    fclk=$(devmem 0xF8000170)
    divider=$(( (33333 * ((io >> 12) & 0x7f) + {rate // 1000} - 1) / {rate // 1000} ))
    echo "FCLK_CLK0 is $(fclk_khz) kHz: setting it to the IO PLL / $divider"
    devmem 0xF8000008 32 0xDF0D
    devmem 0xF8000170 32 $(( (fclk & ~0x03F03F30) | (divider << 8) | (1 << 20) ))
fi
echo "FCLK_CLK0: $(fclk_khz) kHz, built for {rate // 1000} kHz"
echo "kernel AP_CTRL at 0x{base:08X}: $(devmem 0x{base:08X}) (0x00000004: idle, ready to start)"
''', encoding='utf-8', newline='\n')

    archive = run / f'{name}_opu.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in sorted(out.iterdir()):
            zip_file.write(file, f'{name}_opu/{file.name}')
    print(f'\nFor the board (OPU system PetaLinux): {show(archive)}')
    print(f'  Copy the files in {name}_opu/ to {OPU_FIRMWARE}/ on the board, then as root:'
          f'  sh {OPU_FIRMWARE}/load.sh')
    if not dtc:
        print(f'  No dtc here: load.sh compiles {name}.dtsi on the board (the OPU image has dtc).')
    for source in programs:
        if compiler:
            print(f'  Then: {OPU_FIRMWARE}/{source.stem} CUBE.npy SCORES.npy REFERENCE.npy (see {source.name})')
        else:
            print(f'  {source.name} is not built: no ARM compiler (arm-linux-gnueabihf-gcc, or set OPU_CC).'
                  f' Install one (sudo apt install gcc-arm-linux-gnueabihf) and run: python3 main.py pynq --kernel {name}')
    return archive


def package(run, name, board=None):
    """The board's package: the PYNQ zip, or for a board running the OPU system, the OPU one.
    board: the run's, from its run.json when not given."""
    if board is None:
        try:
            board = json.loads((run / 'run.json').read_text(encoding='utf-8')).get('board')
        except (OSError, ValueError):
            board = None
    target = CONFIG['boards'].get(board or CONFIG['board'], {})
    if target.get('software') == 'petalinux':
        return package_opu(run, name)
    return package_pynq(run, name)


def bitstream_runs(name):
    """The runs of a kernel that produced a bitstream, oldest first (folder names are dates)."""
    return sorted(bit.parents[3] for bit in (ROOT / 'build' / name).glob('*/vivado/system.runs/impl_1/*.bit'))


# ------------------------------------------------------------------ The command line

def list_kernels():
    for source_dir in sorted(p for p in (ROOT / 'src' / 'hls').iterdir() if p.is_dir()):
        try:
            kernel = find_kernel(source_dir.name)
        except SystemExit as problem:
            print(f'{source_dir.name}\n  incomplete: {problem}')
            continue
        print(kernel['name'])
        print(f'  top        {kernel["top"]}')
        print(f'  sources    {", ".join(p.name for p in kernel["sources"])}')
        print(f'  testbench  {", ".join(p.name for p in kernel["testbench"])}')
        if kernel['directives']:
            print(f'  directives {show(kernel["directives"])}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=STAGES + ('pynq', 'kernels'))
    parser.add_argument('--kernel', help='folder name in src/hls/')
    parser.add_argument('--skip-cosim', action='store_true', help='ip, bitstream: leave out cosim (slow)')
    parser.add_argument('--dataflow-profile', action='store_true',
                        help='cosim: record when each DATAFLOW process works and stalls, for the '
                             'Dataflow Viewer (slower)')
    parser.add_argument('--clock-ns', type=float,
                        help='clock period target (default: the board\'s "clock_ns" in the config, '
                             f'else {CONFIG["clock_ns"]})')
    parser.add_argument('--run', help='pynq: which build/KERNEL/ folder (default: the newest bitstream)')
    parser.add_argument('--board', choices=sorted(CONFIG['boards']), default=CONFIG['board'],
                        help=f'which FPGA: its part and board script (default {CONFIG["board"]} from the config)')
    args = parser.parse_args()

    if args.stage == 'kernels':
        return list_kernels()
    if not args.kernel:
        parser.error('--kernel is required')
    kernel = find_kernel(args.kernel)

    if args.stage == 'pynq':
        runs = bitstream_runs(args.kernel)
        if args.run:
            runs = [run for run in runs if run.name == args.run]
        if not runs:
            raise SystemExit(f'No bitstream run {args.run or ""} in build/{args.kernel}/')
        return package(runs[-1], args.kernel)

    if args.clock_ns is None:
        args.clock_ns = board_clock_ns(args.board)
    if args.clock_ns <= 0:
        raise SystemExit('--clock-ns must be positive')
    run = new_run_folder(args.kernel, args.stage, args.board)
    part = CONFIG['boards'][args.board]['part']
    print(f'{args.stage} {args.kernel}: {args.board} ({part}), clock {args.clock_ns} ns, '
          f'{job_count()} jobs, in {show(run)}/')
    commit = subprocess.run(['git', 'describe', '--always', '--dirty'], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip() if shutil.which('git') else None
    record = {'kernel': args.kernel, 'stage': args.stage, 'git': commit, 'board': args.board, 'part': part,
              'clock_ns': args.clock_ns, 'skip_cosim': args.skip_cosim, 'dataflow_profile': args.dataflow_profile,
              'started': datetime.datetime.now().isoformat(timespec='seconds'), 'result': 'failed'}
    try:
        if args.stage == 'native':
            native(kernel, run)
        else:
            write_settings(kernel, run, args.stage, args.clock_ns, args.skip_cosim, args.board,
                           args.dataflow_profile)
            shutil.copy2(ROOT / 'scripts' / 'hls.tcl', run)
            hls(run)
            if args.stage == 'bitstream':
                shutil.copy2(ROOT / 'scripts' / 'vivado.tcl', run)
                vivado(run)
        record['result'] = 'passed'
    finally:
        # run.json: what this run was, for comparing runs later; written even if it failed.
        (run / 'run.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        print_estimates(run, kernel['top'])
        print(f'{record["result"].upper()}: {show(run)}/')
    if args.stage == 'ip':
        ip = run / 'hls' / 'solution' / 'impl' / 'ip'
        print(f'\nThe IP, to add to a Vivado IP catalog (Settings > IP > Repository): {show(ip)}/')
        for archive in sorted(ip.glob('*.zip')):
            print(f'  the same as one file: {show(archive)}')
    if args.stage == 'bitstream':
        package(run, args.kernel, args.board)


if __name__ == '__main__':
    main()
