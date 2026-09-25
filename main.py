#!/usr/bin/env python3
"""Portable, dependency-free entry point for native C++, HLS and Vivado.

Run from the repository root: python main.py <stage> [options]
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ElementTree

ROOT = Path(__file__).resolve().parent
STAGES = ('doctor', 'kernels', 'parts', 'native', 'csim', 'csynth', 'cosim',
          'export', 'synth', 'impl', 'bitstream')
SOURCE_PATTERNS = ('*.cpp', '*.cc', '*.cxx')
PARTS_FILE = ROOT / 'config' / 'parts.json'


def tcl_quote(value):
    """One literal Tcl word, including paths containing Tcl metacharacters."""
    value = str(value)
    for old, new in [('\\', '\\\\'), ('"', '\\"'), ('$', '\\$'),
                     ('[', '\\['), (']', '\\]'), ('\n', '\\n'), ('\r', '\\r')]:
        value = value.replace(old, new)
    return '"' + value + '"'


def run_command(command, cwd, log, env=None):
    print('> ' + subprocess.list2cmdline([str(x) for x in command]), flush=True)
    # AMD's Windows launchers are .bat files. Python/Windows invokes cmd.exe
    # for these; the generated script path is passed through the environment.
    with log.open('w', encoding='utf-8') as output:
        with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              encoding='utf-8', errors='replace') as proc:
            for line in proc.stdout:
                print(line, end='', flush=True)
                output.write(line)
                output.flush()
            code = proc.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def compiler_environment(compiler):
    # Keep the selected toolchain's runtime DLLs ahead of other MinGW installs.
    env = os.environ.copy()
    env['PATH'] = str(Path(compiler).resolve().parent) + os.pathsep + env.get('PATH', '')
    return env


def repository_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Project path must stay inside the repository: {value}')
    if not path.exists():
        raise ValueError(f'Project path does not exist: {value}')
    return path


def executable(name, dry=False):
    found = shutil.which(name)
    if found:
        return found
    if dry:
        return name
    raise ValueError(f'Tool not found: {name}. Load AMD settings64.bat/settings64.sh '
                     'in this terminal, or set --hls-tool / --vivado-tool / --cxx.')


def read_config(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return json.loads(path.read_text(encoding='utf-8-sig'))


def part_groups():
    """Named targets from config/parts.json, grouped by device family."""
    if not PARTS_FILE.is_file():
        return {}
    return json.loads(PARTS_FILE.read_text(encoding='utf-8-sig'))


def resolve_part(value):
    """Accept either a board alias from config/parts.json or an exact part."""
    if not value:
        return value
    for group in part_groups().values():
        if value in group:
            return group[value]
    return value


def cpp_files(directory):
    """Repository-relative C++ files of a kernel or testbench directory."""
    found = {p for pattern in SOURCE_PATTERNS for p in directory.glob(pattern)}
    return sorted(p.relative_to(ROOT).as_posix() for p in found)


def discover_kernels(config):
    """Kernels declared in the config plus every src/hls/<name> directory."""
    names = set(config.get('kernels', {}))
    source_root = ROOT / 'src' / 'hls'
    if source_root.is_dir():
        names.update(p.name for p in source_root.iterdir()
                     if p.is_dir() and re.fullmatch(r'[A-Za-z0-9_-]+', p.name))
    return sorted(names)


def resolve_kernel(name, config):
    """Fill a kernel in from the directory layout; config entries take priority."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name or ''):
        raise ValueError('Kernel name must contain only letters, numbers, _ or -')
    kernel = dict(config.get('kernels', {}).get(name) or {})
    source_dir = ROOT / 'src' / 'hls' / name
    testbench_dir = ROOT / 'tb' / name
    if not kernel and not source_dir.is_dir():
        known = ', '.join(discover_kernels(config)) or 'none'
        raise ValueError(f'Unknown kernel {name!r}. Create src/hls/{name}/ and tb/{name}/, '
                         f'or add an entry under "kernels" in the config. Known kernels: {known}')
    kernel.setdefault('top', name)
    kernel.setdefault('sources', cpp_files(source_dir) if source_dir.is_dir() else [])
    kernel.setdefault('testbench', cpp_files(testbench_dir) if testbench_dir.is_dir() else [])
    default_includes = [f'src/hls/{name}'] if source_dir.is_dir() else []
    if (ROOT / 'src' / 'common').is_dir() and 'src/common' not in default_includes:
        default_includes.append('src/common')
    kernel.setdefault('include_dirs', default_includes)
    if 'directives' not in kernel and (ROOT / 'config' / f'{name}.tcl').is_file():
        kernel['directives'] = f'config/{name}.tcl'
    if not kernel['sources']:
        raise ValueError(f'No sources for kernel {name!r}: add C++ files to src/hls/{name}/, '
                         'or list "sources" for it in the config.')
    if not kernel['testbench']:
        raise ValueError(f'No testbench for kernel {name!r}: add a self-checking main() to '
                         f'tb/{name}/, or list "testbench" for it in the config.')
    for field in ('sources', 'testbench', 'include_dirs'):
        for value in kernel.get(field, []):
            repository_path(value)
    if kernel.get('directives'):
        repository_path(kernel['directives'])
    return kernel


def load_config(args):
    config = read_config(args.config)
    name = args.kernel or config.get('default_kernel')
    if not name:
        found = discover_kernels(config)
        if len(found) != 1:
            raise ValueError('Select a kernel with --kernel, or set "default_kernel" in the '
                             f'config. Known kernels: {", ".join(found) or "none"}')
        name = found[0]
    args.kernel = name
    kernel = resolve_kernel(name, config)
    if args.part:
        config['part'] = args.part
    if args.clock_ns is not None:
        config['clock_ns'] = args.clock_ns
    if args.skip_cosim:
        if args.stage in ('native', 'csim', 'csynth', 'cosim'):
            raise ValueError(f'--skip-cosim only applies to export/synth/impl/bitstream, not {args.stage}')
        config['skip_cosim'] = True
    config['part_requested'] = config.get('part')
    config['part'] = resolve_part(config.get('part'))
    if config['part'] and not str(config['part']).lower().startswith('xc'):
        print(f'Warning: {config["part"]!r} is not a name in config/parts.json and does not '
              'look like an AMD part. Run "main.py parts".', file=sys.stderr)
    clock = float(config['clock_ns'])
    if not math.isfinite(clock) or clock <= 0:
        raise ValueError('clock_ns must be finite and positive')
    if not isinstance(config.get('jobs'), int) or config['jobs'] < 1:
        raise ValueError('jobs must be a positive integer')
    if args.stage != 'native' and not config.get('part'):
        raise ValueError('Set the exact FPGA part in config/project.json or use --part.')
    if args.stage in ('impl', 'bitstream') and not config.get('board_script'):
        raise ValueError('A board_script is required for implementation/bitstream. '
                         'See boards/README.md for the board integration contract.')
    if config.get('board_script'):
        repository_path(config['board_script'])
    return config, kernel


def write_settings(run, config, kernel, stage):
    values = dict(root=ROOT.as_posix(), run_dir=run.as_posix(), stage=stage,
                  part=config['part'], clock_ns=config['clock_ns'],
                  jobs=config['jobs'], top=kernel['top'],
                  export_xsa=int(bool(config.get('export_xsa'))),
                  skip_cosim=int(bool(config.get('skip_cosim'))),
                  board_script=(repository_path(config['board_script']).as_posix()
                                if config.get('board_script') else ''),
                  directives=(repository_path(kernel['directives']).as_posix()
                              if kernel.get('directives') else ''))
    lines = [f'set cfg({key}) {tcl_quote(value)}' for key, value in values.items()]
    for field in ('sources', 'testbench', 'include_dirs'):
        paths = [tcl_quote(repository_path(x).as_posix()) for x in kernel.get(field, [])]
        lines.append(f'set cfg({field}) [list {" ".join(paths)}]')
    (run / 'settings.tcl').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_directory(kernel, stage):
    """A run name you can read: build/<kernel>/2026-09-21_11-42-07-csynth/.

    Local time, because the point is finding today's run in a directory
    listing. Two runs in the same second get -2, -3, ... rather than a random
    suffix, so the names still sort in the order the runs happened.
    """
    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base = Path(kernel) / f'{stamp}-{stage}'
    relative, attempt = base, 2
    while (ROOT / 'build' / relative).exists():
        relative = base.with_name(f'{base.name}-{attempt}')
        attempt += 1
    return relative


def point_at_latest(directory):
    """Put a 'latest' symlink beside the run directories, best effort.

    Gives every run a stable path -- reports/conv2d/latest/ -- to tail or open
    without looking up the timestamp. Windows needs developer mode for this, so
    a failure here is not worth failing a run over.
    """
    link = directory.parent / 'latest'
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(directory.name, target_is_directory=True)
    except OSError:
        pass


def collect(run, relative, success):
    reports = ROOT / 'reports' / relative
    reports.mkdir(parents=True, exist_ok=True)
    for file in run.rglob('*'):
        if file.is_file() and file.suffix.lower() in ('.rpt', '.log', '.jou', '.xml'):
            destination = reports / file.relative_to(run)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, destination)
    shutil.copy2(run / 'manifest.json', reports / 'manifest.json')
    point_at_latest(reports)
    if success and (run / 'deliverables').exists():
        artifacts = ROOT / 'artifacts' / relative
        shutil.copytree(run / 'deliverables', artifacts)
        shutil.copy2(run / 'manifest.json', artifacts / 'manifest.json')


def summarize_csynth(run, top):
    """Latency, initiation interval, timing and resources from the HLS report.

    Report element names vary between AMD releases, so every field here is
    optional: an unrecognized report gives an empty summary, never a failed run.
    """
    report = Path(run) / 'hls' / 'solution' / 'syn' / 'report' / f'{top}_csynth.xml'
    if not report.is_file():
        return {}
    root = ElementTree.parse(report).getroot()

    def value(parent, *names):
        for name in names:
            found = parent.findtext(name) if parent is not None else None
            if found and found.strip() not in ('', 'undef'):
                return found.strip()
        return None

    performance = root.find('PerformanceEstimates')
    latency = performance.find('SummaryOfOverallLatency') if performance is not None else None
    timing = performance.find('SummaryOfTimingAnalysis') if performance is not None else None
    assignments = root.find('UserAssignments')
    area = root.find('AreaEstimates')
    used = area.find('Resources') if area is not None else None
    available = area.find('AvailableResources') if area is not None else None
    resources = {}
    for element in (list(used) if used is not None else []):
        count = (element.text or '').strip()
        if count:
            resources[element.tag] = {'used': count,
                                      'available': value(available, element.tag)}
    summary = {
        'latency_min': value(latency, 'Best-caseLatency'),
        'latency_max': value(latency, 'Worst-caseLatency'),
        'interval_min': value(latency, 'Interval-min'),
        'interval_max': value(latency, 'Interval-max'),
        # 2023.1 keeps the target under UserAssignments, not with the estimate.
        'clock_target': value(timing, 'TargetClockPeriod') or value(assignments, 'TargetClockPeriod'),
        'clock_estimate': value(timing, 'EstimatedClockPeriod'),
        'clock_uncertainty': value(assignments, 'ClockUncertainty'),
        'clock_unit': value(timing, 'unit') or 'ns',
        'part': value(assignments, 'Part'),
        'resources': resources or None,
    }
    summary = {key: item for key, item in summary.items() if item}
    return summary if len(summary) > 1 else {}


def print_csynth_summary(summary):
    def span(low, high):
        if not low and not high:
            return None
        return low if low == high or not high else f'{low} - {high}'

    unit = summary.get('clock_unit', 'ns')
    lines = []
    cycles = span(summary.get('latency_min'), summary.get('latency_max'))
    interval = span(summary.get('interval_min'), summary.get('interval_max'))
    if cycles:
        lines.append(f'  latency   {cycles} cycles')
    if interval:
        lines.append(f'  II        {interval} cycles')
    if summary.get('clock_estimate'):
        target, uncertainty = summary.get('clock_target'), summary.get('clock_uncertainty')
        budget = f' (target {target} {unit}' if target else ''
        if budget and uncertainty:
            budget += f', uncertainty {uncertainty} {unit}'
        lines.append(f'  timing    {summary["clock_estimate"]} {unit}{budget})' if budget
                     else f'  timing    {summary["clock_estimate"]} {unit}')
    order = ('DSP', 'DSP48E', 'BRAM_18K', 'URAM', 'FF', 'LUT')
    resources = summary.get('resources') or {}
    names = [n for n in order if n in resources] + [n for n in resources if n not in order]
    counts = [f'{name} {resources[name]["used"]}'
              + (f'/{resources[name]["available"]}' if resources[name].get('available') else '')
              for name in names]
    if counts:
        lines.append('  resources ' + '  '.join(counts))
    if lines:
        print('HLS synthesis estimates:')
        print('\n'.join(lines))


def doctor(args):
    print(f'Python: {sys.version.split()[0]} ({platform.system()})')
    names = ['git', args.cxx or os.environ.get('CXX', 'g++'),
             args.hls_tool or 'vitis-run', 'vitis_hls', args.vivado_tool or 'vivado']
    for name in dict.fromkeys(names):
        print(f'{name}: {shutil.which(name) or "NOT FOUND"}')
    print('Discovery only; does not validate licenses, device support or tool compatibility.')


def parts(args):
    """Named targets you can pass to --part instead of an exact part string."""
    groups = part_groups()
    if not groups:
        print(f'No named targets; add them to {PARTS_FILE.relative_to(ROOT).as_posix()}.')
        return
    width = max(len(alias) for group in groups.values() for alias in group)
    for group, entries in groups.items():
        print(group)
        for alias, part in entries.items():
            print(f'  {alias.ljust(width)}  {part}')
    print('\nAny exact part string is accepted too. These are common boards, not a '
          'validated list:\nconfirm yours in your own install with '
          '"vivado -mode tcl", then get_parts -filter {NAME =~ xc7z020*}.')


def kernels(args):
    """Show what the runner found, so file discovery is never a guessing game."""
    config = read_config(args.config)
    found = discover_kernels(config)
    if not found:
        print('No kernels found. Create src/hls/<name>/ and tb/<name>/ to add one.')
        return
    default = config.get('default_kernel') or (found[0] if len(found) == 1 else None)
    for name in found:
        print(name + ('  (default)' if name == default else ''))
        try:
            kernel = resolve_kernel(name, config)
        except ValueError as error:
            print(f'  incomplete: {error}')
            continue
        print(f'  top        {kernel["top"]}')
        print(f'  sources    {", ".join(kernel["sources"])}')
        print(f'  testbench  {", ".join(kernel["testbench"])}')
        print(f'  includes   {", ".join(kernel["include_dirs"]) or "-"}')
        print(f'  directives {kernel.get("directives") or "-"}')


def build_parser():
    """The command line, in one place: tabcompletion.py reads it from here."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--kernel', help='Kernel name; defaults to default_kernel in the config')
    parser.add_argument('--config', default='config/project.json')
    parser.add_argument('--part', help='Exact AMD device part, or a name from '
                                       '"main.py parts" (config/parts.json)')
    parser.add_argument('--clock-ns', type=float)
    parser.add_argument('--hls-tool', help='vitis-run or vitis_hls executable/path')
    parser.add_argument('--vivado-tool', default='vivado')
    parser.add_argument('--cxx', help='GCC/Clang C++ executable/path (or CXX environment variable)')
    parser.add_argument('--skip-cosim', action='store_true',
                        help='Skip C/RTL co-simulation before export/synth/impl/bitstream')
    parser.add_argument('--dry-run', action='store_true', help='Show plan without tools or writes')
    return parser


def main():
    args = build_parser().parse_args()
    if args.stage == 'doctor':
        doctor(args)
        return 0
    if args.stage == 'kernels':
        kernels(args)
        return 0
    if args.stage == 'parts':
        parts(args)
        return 0
    config, kernel = load_config(args)
    commands = []
    if args.stage == 'native':
        cxx = executable(args.cxx or os.environ.get('CXX', 'g++'), args.dry_run)
        commands.append(('compile', [cxx, '-std=c++14', '-O2', '-Wall', '-Wextra',
                         '-Wno-unknown-pragmas', '-Wno-unused-label',
                         *['-I' + str(repository_path(x)) for x in kernel['include_dirs']],
                         *[str(repository_path(x)) for x in kernel['sources'] + kernel['testbench']],
                         '-o', 'testbench.exe' if os.name == 'nt' else 'testbench']))
    else:
        hls = args.hls_tool or ('vitis-run' if shutil.which('vitis-run') else
                               'vitis_hls' if shutil.which('vitis_hls') else 'vitis-run')
        hls_exe = executable(hls, args.dry_run)
        options = ['-f'] if 'vitis_hls' in Path(hls).name.lower() else ['--mode', 'hls', '--tcl']
        commands.append(('hls', [hls_exe, *options, 'hls.tcl']))
        if args.stage in ('synth', 'impl', 'bitstream'):
            commands.append(('vivado', [executable(args.vivado_tool, args.dry_run),
                             '-mode', 'batch', '-source', 'vivado.tcl',
                             '-notrace', '-log', 'vivado.log', '-journal', 'vivado.jou']))
    alias = config.get('part_requested')
    named = f' ({alias})' if alias and alias != config.get('part') else ''
    print(f'Stage: {args.stage}; kernel: {args.kernel}; part: {config.get("part")}{named}; '
          f'clock: {config["clock_ns"]} ns')
    if args.dry_run:
        for name, command in commands:
            print(f'{name}: {subprocess.list2cmdline(command)}')
        if args.stage == 'native':
            print('Then execute the compiled testbench.')
        else:
            cosim = ('skipped (--skip-cosim)' if config.get('skip_cosim')
                     else 'for cosim/export/synth/impl/bitstream')
            print('HLS prerequisites: C simulation; synthesis unless csim; '
                  f'RTL co-simulation {cosim}; '
                  'IP export for export/synth/impl/bitstream.')
        return 0
    relative = run_directory(args.kernel, args.stage)
    run = ROOT / 'build' / relative
    run.mkdir(parents=True)
    point_at_latest(run)
    manifest = {'stage': args.stage, 'config': config, 'kernel': args.kernel,
                'kernel_files': kernel, 'commands': commands,
                'platform': platform.platform(), 'python': sys.version,
                'started': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                'status': 'running', 'run': relative.as_posix()}
    if shutil.which('git'):
        for label, argv in [('git_commit', ['rev-parse', 'HEAD']),
                            ('git_status', ['status', '--porcelain'])]:
            info = subprocess.run(['git', *argv], cwd=ROOT, capture_output=True, text=True)
            manifest[label] = info.stdout.strip() if info.returncode == 0 else None
    hashed = [ROOT / 'main.py']
    for folder in ('src', 'tb', 'config', 'scripts', 'boards', 'constraints'):
        hashed += [p for p in (ROOT / folder).rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts]
    manifest['source_sha256'] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in hashed
    }
    status = 1
    try:
        if args.stage != 'native':
            write_settings(run, config, kernel, args.stage)
            for script in ('hls.tcl', 'vivado.tcl'):
                shutil.copy2(ROOT / 'scripts' / script, run / script)
        for name, command in commands:
            env = compiler_environment(command[0]) if args.stage == 'native' else None
            run_command(command, run, run / f'{name}-console.log', env=env)
        if args.stage == 'native':
            binary = run / ('testbench.exe' if os.name == 'nt' else 'testbench')
            run_command([str(binary)], run, run / 'testbench.log',
                        env=compiler_environment(commands[0][1][0]))
        status = 0
    finally:
        if args.stage not in ('native', 'csim'):
            try:
                summary = summarize_csynth(run, kernel['top'])
            except Exception as error:  # an unexpected report layout must not fail the run
                summary = {}
                print(f'Could not read the HLS synthesis report ({error}).')
            if summary:
                manifest['csynth'] = summary
                print_csynth_summary(summary)
            else:
                print(f'No synthesis estimates parsed; see {run / "hls"} for the raw reports.')
        manifest['status'] = 'passed' if status == 0 else 'failed'
        (run / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        collect(run, relative, status == 0)
        reports = ROOT / 'reports' / relative
        print(f'Build:   {run}')
        print(f'Reports: {reports}   (also reports/{args.kernel}/latest)')
        named = {'console': reports / f'{commands[-1][0]}-console.log',
                 'synthesis': (reports / 'hls' / 'solution' / 'syn' / 'report'
                               / f'{kernel["top"]}_csynth.rpt'),
                 'cosim': (reports / 'hls' / 'solution' / 'sim' / 'report'
                           / f'{kernel["top"]}_cosim.rpt')}
        for label, file in named.items():
            if file.is_file():
                print(f'  {label:<10} {file.relative_to(ROOT).as_posix()}')
        if status == 0 and (run / 'deliverables').exists():
            print(f'Artifacts: {ROOT / "artifacts" / relative}')
    return status


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(2)
    except subprocess.CalledProcessError as error:
        print(f'Tool failed with exit code {error.returncode}; see the run logs.', file=sys.stderr)
        sys.exit(error.returncode if 0 < error.returncode < 256 else 1)
