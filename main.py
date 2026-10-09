#!/usr/bin/env python3
"""Build an HLS kernel: from C++ to a ZCU104 bitstream and a zip for the PYNQ notebook.

    python3 main.py STAGE --kernel KERNEL

Each stage includes the ones above it:

    native     compile src/hls/KERNEL/ + tb/KERNEL/ with g++ and run the testbench
    csim       the same testbench, inside Vitis HLS                     (scripts/hls.tcl)
    csynth     ... then C++ -> Verilog
    cosim      ... then simulate that Verilog with the testbench
    bitstream  ... then export the IP; the block design around it, synthesis, place and
               route, .bit (scripts/vivado.tcl + boards/BOARD/system.tcl); the PYNQ zip

and two helpers:

    pynq       re-make the PYNQ zip of an earlier bitstream run (--run, default the newest)
    kernels    list the kernels and the files found for each

--board picks the FPGA (its part and board script, from config/project.json); the
default is the config's "board". Every run gets its own folder, build/KERNEL/DATE-STAGE/
(DATE-STAGE-BOARD for a board other than the default), with everything it produced in it.
build/KERNEL/latest points at the newest one. Settings are in config/project.json.
"""
import argparse
import datetime
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
STAGES = ('native', 'csim', 'csynth', 'cosim', 'bitstream')


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


def write_settings(kernel, run, stage, clock_ns, skip_cosim, board=None):
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
        'skip_cosim': int(skip_cosim), 'board_script': ROOT / target['script'],
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
    notebook = ROOT / 'software' / 'pynq' / name

    out = run / f'{name}_pynq'
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    shutil.copy2(bit, out / f'{name}.bit')  # PYNQ finds the .hwh by the .bit's name
    shutil.copy2(hwh, out / f'{name}.hwh')
    shutil.copy2(header, out / header.name)
    for file in (notebook.glob('*') if notebook.is_dir() else []):
        if file.is_file() and file.name != 'include.txt':
            shutil.copy2(file, out / file.name)
    include = notebook / 'include.txt'  # other repository files the notebook uses, one per line
    if include.is_file():
        for line in include.read_text(encoding='utf-8').splitlines():
            line = line.split('#')[0].strip()
            if line:
                shutil.copy2(ROOT / line, out / Path(line).name)

    archive = run / f'{name}_pynq.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in sorted(out.iterdir()):
            zip_file.write(file, f'{name}_pynq/{file.name}')
    print(f'\nFor the board: {show(archive)}')
    print(f'  Upload it in Jupyter (http://BOARD:9090), run  !unzip -o {name}_pynq.zip  in a cell,')
    print(f'  then open {name}_pynq/{name}.ipynb and run it from the top.')
    return archive


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
    parser.add_argument('--skip-cosim', action='store_true', help='bitstream: leave out cosim (slow)')
    parser.add_argument('--clock-ns', type=float, default=CONFIG['clock_ns'],
                        help=f'clock period target (default {CONFIG["clock_ns"]} from the config)')
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
        return package_pynq(runs[-1], args.kernel)

    if args.clock_ns <= 0:
        raise SystemExit('--clock-ns must be positive')
    run = new_run_folder(args.kernel, args.stage, args.board)
    part = CONFIG['boards'][args.board]['part']
    print(f'{args.stage} {args.kernel}: {args.board} ({part}), clock {args.clock_ns} ns, '
          f'{job_count()} jobs, in {show(run)}/')
    commit = subprocess.run(['git', 'describe', '--always', '--dirty'], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip() if shutil.which('git') else None
    record = {'kernel': args.kernel, 'stage': args.stage, 'git': commit, 'board': args.board, 'part': part,
              'clock_ns': args.clock_ns, 'skip_cosim': args.skip_cosim,
              'started': datetime.datetime.now().isoformat(timespec='seconds'), 'result': 'failed'}
    try:
        if args.stage == 'native':
            native(kernel, run)
        else:
            write_settings(kernel, run, args.stage, args.clock_ns, args.skip_cosim, args.board)
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
    if args.stage == 'bitstream':
        package_pynq(run, args.kernel)


if __name__ == '__main__':
    main()
