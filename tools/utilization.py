#!/usr/bin/env python3
"""Every kernel's routed utilization in one page, from Vivado's reports (no third-party packages).

    python tools/utilization.py [FOLDER]

FOLDER (default optimizations_implemented/utilization/) holds K_utilization.rpt for each kernel K:
build/K/RUN/reports/impl/utilization.rpt (report_utilization -hierarchical, scripts/vivado.tcl),
renamed by kernel, and sources.txt, the run each came from ("K <- build/K/RUN/...").

Writes optimizations_implemented/utilization.md: per board, the whole design and the kernel
against the chip, then where each kernel's resources go. In the other files there, it fills the
block between "<!-- utilization.py: MODEL ... -->" and "<!-- /utilization.py -->" with a table
of those models' kernels, so the model files stay in step with the reports.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'optimizations_implemented'
FOLDER = DOCS / 'utilization'
OUT = DOCS / 'utilization.md'

sys.path.insert(0, str(ROOT / 'tools'))
from models import MODELS  # noqa: E402

COLUMNS = ['LUTs', 'FFs', 'BRAM36', 'URAM', 'DSPs']
# What each chip has, by the start of its part name.
DEVICES = {
    'xczu7ev': ('ZCU104 (ZU7EV)', {'LUTs': 230400, 'FFs': 460800, 'BRAM36': 312, 'URAM': 96, 'DSPs': 1728}),
    'xc7z030': ('Zynq-7030', {'LUTs': 78600, 'FFs': 157200, 'BRAM36': 265, 'URAM': 0, 'DSPs': 400}),
    'xc7z020': ('Zynq-7020 (ZedBoard, PYNQ-Z2)', {'LUTs': 53200, 'FFs': 106400, 'BRAM36': 140, 'URAM': 0, 'DSPs': 220}),
}
# Kernels whose source is gone but whose build is worth keeping.
REMOVED = {'sp_unet_small_opt': 'Winograd, removed 2026-10-31: see sp_unet.md'}
# Modules renamed or merged since the reports were made: old name -> today's.
RENAMED = {
    'conv3x3_grouped': 'conv3x3_tiled', 'conv3x3_grouped_patch': 'conv3x3_tiled_patch',
    'conv3x3_serial': 'conv3x3_tiled', 'conv3x3_serial_patch': 'conv3x3_tiled_patch',
    'conv3x3_lines': 'conv3x3_tiled_ram', 'conv3x3_lines_patch': 'conv3x3_tiled_ram_patch',
    'conv3x3_packed_layer': 'conv3x3_2mul', 'conv3x3_wide': 'conv3x3_2mul', 'conv3x3_packed': 'conv3x3_2mul_patch',
    'conv3x3_narrow': 'conv3x3_tiled', 'conv3x3_stream': 'conv3x3_tiled_patch',
    'conv3x3_all_in': 'conv3x3_tiled', 'conv3x3_all_in_patch': 'conv3x3_tiled_patch',
    'pointwise_wide': 'pointwise_packed', 'pointwise': 'conv1x1', 'leaky_head': 'conv1x1', 'leaky_conv1x1': 'conv1x1', 'pointwise1d': 'conv1x1',
    'conv1d_engine': 'conv1d', 'conv1d_stream': 'conv1d', 'conv1d_map': 'conv1d', 'conv1d_positions': 'conv1d',
    'dense_scores': 'dense_map', 'dense_stream': 'dense_positions',
    'pool_assemble': 'pool_to_map', 'pool_pairs': 'pool1d', 'pool_adjacent': 'pool1d', 'pool1d_wide': 'pool1d',
    'read_rows_ahead': 'read_patch_rows', 'read_pixels_ahead': 'read_pixels', 'patch_buffer_run': 'patch_buffer_body',
}
# HLS's operator instances in the golden kernels, by their short name.
OPERATORS = {'fmul': 'float multipliers (fmul)', 'fadd': 'float adders (fadd)',
             'mul': 'integer multipliers (mul)'}
FIFOS = 'FIFOs and memories'
AXI = 'AXI ports (m_axi, s_axi)'
GLUE = 'glue (the function itself)'
OUTSIDE = 'the rest: ports, control, top-level arrays'
OTHERS = 'others'
LAST = (OTHERS, FIFOS, AXI, GLUE, OUTSIDE)
# A part smaller than this share of the kernel's LUTs, with no DSP or block RAM, goes into OTHERS.
SMALL = 0.01
BLOCK = re.compile(r'(<!-- utilization\.py: ([\w ]+) -->\n).*?(<!-- /utilization\.py -->)', re.S)


class Node:
    def __init__(self, name, depth, use):
        self.name, self.depth, self.use = name, depth, use
        self.children = []


def parse(text):
    """The report's hierarchy table -> (part name, the top Node)."""
    part = re.search(r'^\| Device\s*:\s*(\S+)', text, re.M).group(1)
    header, stack, top = None, [], None
    for line in text.splitlines():
        cells = line.split('|')[1:-1] if line.startswith('|') else []
        if not cells:  # also the header block's "| Device : ..." lines
            continue
        if cells[0].strip() == 'Instance':
            header = [c.strip() for c in cells]
            continue
        if header is None:
            continue
        raw = dict(zip(header, (c.strip() for c in cells)))
        use = {
            'LUTs': int(raw['Total LUTs']),
            'FFs': int(raw['FFs']),
            # A RAMB18 is half a BRAM36, as Vivado counts them in its summary.
            'BRAM36': int(raw['RAMB36']) + int(raw['RAMB18']) / 2,
            'URAM': int(raw.get('URAM', 0)),  # the 7-series reports have no URAM column
            'DSPs': int(raw['DSP Blocks']),
        }
        depth = len(cells[0]) - len(cells[0].lstrip())
        node = Node(cells[0].strip(), depth, use)
        while stack and stack[-1].depth >= depth:
            stack.pop()
        if stack:
            stack[-1].children.append(node)
        else:
            top = node
        stack.append(node)
    return part, top


def find_kernel(top):
    """The kernel's own instance: the block design's `kernel`."""
    stack = [top]
    while stack:
        node = stack.pop()
        if node.name == 'kernel':
            return node
        stack.extend(node.children)
    raise ValueError('no instance named kernel')


def short_name(instance):
    """grp_conv3x3_2mul_patch_32_32_112_6_8_16_2_s_fu_548 -> conv3x3_2mul_patch: drop HLS's
    prefixes, suffixes and template arguments (they start at the first _<digit>), and give
    a module renamed since its report today's name."""
    name = re.sub(r'^grp_', '', instance)
    name = re.sub(r'(_fu_\d+|_U0|_U)$', '', name)
    # The arguments start at the first _<digit>; conv3x3_2mul's own _2 is not one.
    name = re.sub(r'_(?!2mul)\d.*$', '', name)
    name = re.sub(r'_s$', '', name)
    return RENAMED.get(name, name)


def parts(kernel):
    """Where the kernel's resources go: [(label, copies, use)], largest first."""
    node = kernel
    # Step into wrappers (HLS's inst, a DATAFLOW region, the golden function) that hold nearly
    # everything, to show what is inside them.
    while True:
        big = [c for c in node.children
               if not c.name.startswith('(') and c.children and c.use['LUTs'] >= 0.8 * node.use['LUTs']]
        if len(big) != 1:
            break
        node = big[0]
    groups = {}

    def add(label, use):
        copies, sums = groups.get(label, (0, dict.fromkeys(COLUMNS, 0)))
        groups[label] = (copies + 1, {k: sums[k] + use[k] for k in COLUMNS})

    for child in node.children:
        name = child.name
        if name.startswith('('):
            add(GLUE, child.use)
        elif re.search(r'_[ms]_axi_U$', name):
            add(AXI, child.use)
        elif name.endswith('_U'):
            add(FIFOS, child.use)
        elif (child.use['LUTs'] < SMALL * kernel.use['LUTs']
              and not child.use['DSPs'] and not child.use['BRAM36']):
            add(OTHERS, child.use)
        else:
            label = short_name(name)
            add(OPERATORS.get(label, label), child.use)
    # What is outside the function stepped into.
    outside = {k: kernel.use[k] - node.use[k] for k in COLUMNS}
    if any(outside.values()):
        add(OUTSIDE, outside)
    rows = [(label, copies, use) for label, (copies, use) in groups.items()]
    rows.sort(key=lambda r: (LAST.index(r[0]) if r[0] in LAST else -1, -r[2]['DSPs'], -r[2]['LUTs']))
    return rows


def number(value):
    text = f'{value:,.1f}' if value != int(value) else f'{int(value):,}'
    return text.replace(',', ' ')


def share(value, available):
    if not available:
        return '–'
    if not value:
        return '0'
    return f'{number(value)} ({100 * value / available:.1f} %)'


def model_of(kernel):
    """The model a kernel belongs to: the longest model name it starts with (tools/models.py)."""
    names = sorted(MODELS, key=len, reverse=True)
    return next((m for m in names if kernel == m or kernel.startswith(m + '_')), None)


def order(kernel):
    """By model in tools/models.py's order, the golden kernel first."""
    model = model_of(kernel)
    position = list(MODELS).index(model) if model else len(MODELS)
    return (position, kernel != model, kernel)


def read_reports(folder):
    """[(kernel, board, run, whole design, kernel Node)] in model order."""
    runs = {}
    sources = folder / 'sources.txt'
    if sources.exists():
        for line in sources.read_text().splitlines():
            if '<-' in line:
                kernel, report = (s.strip() for s in line.split('<-'))
                runs[kernel] = report.split('/')[2]
    rows = []
    for path in folder.glob('*_utilization.rpt'):
        kernel = path.name[:-len('_utilization.rpt')]
        part, top = parse(path.read_text())
        board = next(k for k in DEVICES if part.startswith(k))
        rows.append((kernel, board, runs.get(kernel, ''), top, find_kernel(top)))
    return sorted(rows, key=lambda r: order(r[0]))


def label(kernel):
    return f'`{kernel}`' + (f' ({REMOVED[kernel]})' if kernel in REMOVED else '')


def page(rows):
    out = ['# Utilization (routed)', '',
           'Every kernel\'s resources after place and route, from `reports/impl/utilization.rpt` of its',
           'newest bitstream build, `build/K/RUN/` (RUN in the tables). The reports themselves are in',
           '[`utilization/`](utilization/). Written by `python tools/utilization.py`: do not edit by hand.',
           'BRAM36 counts a RAMB18 as half. These are the real numbers; csynth\'s estimates are not here.',
           '']
    for board, (title, chip) in DEVICES.items():
        mine = [r for r in rows if r[1] == board]
        if not mine:
            continue
        columns = [c for c in COLUMNS if chip[c]]
        out += [f'## {title}', '',
                'The chip has ' + ', '.join(f'{number(chip[c])} {c}' for c in columns) + '.', '']
        for what, pick in (('Kernel alone', 4), ('Whole design: the kernel, interconnects and reset', 3)):
            out += [f'### {what}', '',
                    '| Kernel | Run | ' + ' | '.join(columns) + ' |',
                    '|---|---|' + '---|' * len(columns)]
            for row in mine:
                use = row[pick].use
                out.append(f'| {label(row[0])} | {row[2]} | '
                           + ' | '.join(share(use[c], chip[c]) for c in columns) + ' |')
            out.append('')
        out += ['### Where it goes', '',
                'Each kernel\'s processes (inside its DATAFLOW region or golden function, when nearly',
                'all of it is there). Copies of one module are summed (× n); parts under 1 % of the',
                f'kernel\'s LUTs with no DSP or block RAM are summed into "{OTHERS}".', '']
        for kernel, _, _, _, node in mine:
            out += [f'#### `{kernel}`', '',
                    '| Part | ' + ' | '.join(columns) + ' |',
                    '|---|' + '---|' * len(columns)]
            for part, copies, use in parts(node):
                name = part if copies == 1 or part in LAST else f'{part} × {copies}'
                out.append(f'| {name} | ' + ' | '.join(number(use[c]) for c in columns) + ' |')
            out.append('| **total** | ' + ' | '.join(f'**{number(node.use[c])}**' for c in columns) + ' |')
            out.append('')
    return '\n'.join(out)


def block(rows, models):
    """The table for one model file: its kernels, alone, against their chip."""
    mine = [r for r in rows if model_of(r[0]) in models]
    out = ['Routed, kernel alone (without the interconnects); by process in',
           '[utilization.md](utilization.md). Written by `python tools/utilization.py`.', '',
           '| Kernel | Board | ' + ' | '.join(COLUMNS) + ' |',
           '|---|---|' + '---|' * len(COLUMNS)]
    for kernel, board, _, _, node in mine:
        chip = DEVICES[board][1]
        out.append(f'| {label(kernel)} | {DEVICES[board][0].split(" (")[0]} | '
                   + ' | '.join(share(node.use[c], chip[c]) for c in COLUMNS) + ' |')
    if not mine:
        out.append('| (no report yet) | | | | | | |')
    return '\n'.join(out) + '\n'


def fill_blocks(rows):
    changed = []
    for path in sorted(DOCS.glob('*.md')):
        text = path.read_text(encoding='utf-8')
        new = BLOCK.sub(lambda m: m.group(1) + block(rows, m.group(2).split()) + m.group(3), text)
        if new != text:
            path.write_bytes(new.encode('utf-8'))  # LF, also on Windows
            changed.append(path.name)
    return changed


def main(argv):
    folder = Path(argv[1]) if len(argv) > 1 else FOLDER
    rows = read_reports(folder)
    OUT.write_bytes((page(rows) + '\n').encode('utf-8'))
    print(f'wrote {OUT.relative_to(ROOT)} ({len(rows)} reports)')
    for name in fill_blocks(rows):
        print(f'updated optimizations_implemented/{name}')


if __name__ == '__main__':
    main(sys.argv)
