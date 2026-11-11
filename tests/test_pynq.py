"""The PYNQ notebooks: valid, and their code parses.

They run end to end against a fake board in test_notebooks.py; what goes into the package
next to them (scripts/pynq.sh) is checked in test_scripts.py.
"""
import ast
import json
import unittest

from helpers import ROOT


class NotebookFileTests(unittest.TestCase):
    def test_every_notebook_is_valid_and_its_code_parses(self):
        notebooks = sorted((ROOT / 'software/pynq').glob('**/*.ipynb'))
        self.assertEqual([p.name for p in notebooks], ['benchmark.ipynb', 'classify.ipynb', 'classify_golden.ipynb'],
                         'the two shared notebooks and the benchmark')
        for path in notebooks:
            with self.subTest(notebook=path.relative_to(ROOT).as_posix()):
                notebook = json.loads(path.read_text(encoding='utf-8'))
                self.assertEqual(notebook['nbformat'], 4)
                code = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
                self.assertTrue(code)
                for cell in code:
                    if not cell.lstrip().startswith(('!', '%')):  # shell and magic lines are not Python
                        ast.parse(cell)


if __name__ == '__main__':
    unittest.main()
