"""tools/utilization.py: reading Vivado's hierarchical utilization reports into the docs."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import utilization  # noqa: E402

COLUMNS = ('| {:<40} | {:>10} | {:>10} | {:>7} | {:>4} | {:>5} | {:>6} | {:>6} | {:>4} | {:>10} |')


def row(name, depth, luts, ffs, ramb36=0, ramb18=0, uram=0, dsps=0):
    return COLUMNS.format(' ' * depth + name, luts, luts, 0, 0, ffs, ramb36, ramb18, uram, dsps)


# A small report in Vivado's layout: two copies of one process, a FIFO, the AXI ports.
REPORT = '\n'.join([
    '| Device       : xczu7ev-ffvc1156-2-e',
    '| Design State : Routed',
    '+---+',
    COLUMNS.replace('{:<40}', '{:^40}').format('Instance', 'Total LUTs', 'Logic LUTs', 'LUTRAMs',
                                               'SRLs', 'FFs', 'RAMB36', 'RAMB18', 'URAM', 'DSP Blocks'),
    '+---+',
    row('system_wrapper', 0, 12000, 9000, 3, 1, 2, 100),
    row('system_i', 2, 12000, 9000, 3, 1, 2, 100),
    row('ctrl', 4, 2000, 1000),
    row('kernel', 4, 10000, 8000, 3, 1, 2, 100),
    row('inst', 6, 10000, 8000, 3, 1, 2, 100),
    row('(inst)', 8, 50, 10),
    row('control_s_axi_U', 8, 400, 300),
    row('gmem0_m_axi_U', 8, 550, 690, 2),
    row('conv_32_8_s_fu_10', 8, 4000, 3000, 0, 1, 0, 60),
    row('mul_U1', 10, 100, 0, 0, 0, 0, 60),
    row('conv_32_8_U0', 8, 4000, 3000, 0, 0, 0, 40),
    row('tiny_U0', 8, 50, 100),
    row('s_a_U', 8, 950, 900, 1, 0, 2, 0),
    '+---+',
])


class UtilizationTests(unittest.TestCase):
    def test_parse_reads_the_hierarchy_and_counts_ramb18_as_half(self):
        part, top = utilization.parse(REPORT)
        self.assertEqual(part, 'xczu7ev-ffvc1156-2-e')
        kernel = utilization.find_kernel(top)
        self.assertEqual(kernel.use, {'LUTs': 10000, 'FFs': 8000, 'BRAM36': 3.5, 'URAM': 2, 'DSPs': 100})
        self.assertEqual([c.name for c in top.children[0].children], ['ctrl', 'kernel'])

    def test_parts_sum_copies_and_group_ports_fifos_and_small_ones(self):
        _, top = utilization.parse(REPORT)
        parts = {label: (copies, use) for label, copies, use in utilization.parts(utilization.find_kernel(top))}
        self.assertEqual(parts['conv'][0], 2)
        self.assertEqual(parts['conv'][1]['DSPs'], 100)
        self.assertEqual(parts[utilization.AXI][1]['LUTs'], 950)
        self.assertEqual(parts[utilization.FIFOS][1]['URAM'], 2)
        self.assertEqual(parts[utilization.OTHERS][1]['LUTs'], 50)
        self.assertEqual(sum(use['LUTs'] for _, use in parts.values()), 10000)

    def test_short_names_drop_hls_decoration(self):
        for instance, name in [('grp_conv3x3_2mul_patch_32_32_112_6_8_16_2_s_fu_548', 'conv3x3_2mul_patch'),
                               ('grp_conv3x3_packed_32_32_112_6_8_16_2_s_fu_548', 'conv3x3_2mul_patch'),
                               ('read_rows_ahead_32_32_30_2_0_8_2_vector_unsigned_short_4ul_U0', 'read_patch_rows'),
                               ('fmul_32ns_32ns_32_4_max_dsp_1_U70', 'fmul'),
                               ('patch_buffer_U0', 'patch_buffer')]:
            self.assertEqual(utilization.short_name(instance), name)

    def test_blocks_are_filled_between_their_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'justoliunet_opt_utilization.rpt').write_text(REPORT)
            rows = utilization.read_reports(folder)
        text = 'before\n<!-- utilization.py: justoliunet -->\nold\n<!-- /utilization.py -->\nafter\n'
        new = utilization.BLOCK.sub(
            lambda m: m.group(1) + utilization.block(rows, m.group(2).split()) + m.group(3), text)
        self.assertIn('| `justoliunet_opt` | ZCU104 | 10 000 (4.3 %)', new)
        self.assertNotIn('old', new)
        self.assertTrue(new.startswith('before\n') and new.endswith('<!-- /utilization.py -->\nafter\n'))

    def test_the_docs_match_the_reports(self):
        """utilization.md and the model files' tables are generated: run python tools/utilization.py
        after adding reports."""
        rows = utilization.read_reports(utilization.FOLDER)
        if not rows:
            self.skipTest('no reports in optimizations_implemented/utilization/')
        self.assertEqual(utilization.OUT.read_text(encoding='utf-8'), utilization.page(rows) + '\n')
        for path in utilization.DOCS.glob('*.md'):
            text = path.read_text(encoding='utf-8')
            for m in utilization.BLOCK.finditer(text):
                self.assertEqual(m.group(0), m.group(1) + utilization.block(rows, m.group(2).split()) + m.group(3),
                                 f'{path.name}: run python tools/utilization.py')


if __name__ == '__main__':
    unittest.main()
