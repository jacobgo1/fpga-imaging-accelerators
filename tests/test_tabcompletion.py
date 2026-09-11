"""Drive tabcompletion.sh through a real bash, so drift from main.py is caught.

Skipped where bash is unavailable; the completion itself is bash-only by design.
"""
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which('bash')


@unittest.skipIf(BASH is None, 'tabcompletion.sh needs bash (Git Bash, WSL or Linux)')
class CompletionTests(unittest.TestCase):
    def complete(self, *words, cword=None):
        """Return what pressing TAB after `words` would offer."""
        cword = len(words) - 1 if cword is None else cword
        quoted = ' '.join(f'"{w}"' for w in words)
        script = (f'source "{ROOT.as_posix()}/tabcompletion.sh"\n'
                  f'COMP_WORDS=({quoted})\n'
                  f'COMP_CWORD={cword}\n'
                  'COMPREPLY=()\n'
                  '_main_completion\n'
                  'printf "%s\\n" "${COMPREPLY[@]}"\n')
        result = subprocess.run([BASH, '-c', script], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return [line for line in result.stdout.split() if line]

    def test_stages_are_offered_first(self):
        found = self.complete('main', '')
        for stage in ('doctor', 'kernels', 'parts', 'native', 'csim', 'csynth',
                      'cosim', 'export', 'synth', 'impl', 'bitstream'):
            self.assertIn(stage, found)

    def test_stages_come_from_the_runner(self):
        """Read from main.py --help, never copied into the shell script."""
        import importlib.util
        spec = importlib.util.spec_from_file_location('runner', ROOT / 'main.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        self.assertEqual(sorted(self.complete('main', '')), sorted(runner.STAGES))

    def test_prefix_filters_stages(self):
        self.assertEqual(sorted(self.complete('main', 'cs')), ['csim', 'csynth'])

    def test_flags_offered_after_a_stage(self):
        found = self.complete('main', 'csynth', '--')
        self.assertIn('--part', found)
        self.assertIn('--kernel', found)
        self.assertIn('--dry-run', found)

    def test_used_flag_is_not_offered_again(self):
        found = self.complete('main', 'csynth', '--part', 'zcu104', '--')
        self.assertNotIn('--part', found)
        self.assertIn('--kernel', found)

    def test_part_offers_board_names(self):
        found = self.complete('main', 'csynth', '--part', '')
        self.assertIn('zcu104', found)
        self.assertIn('zybo-z7-20', found)
        self.assertNotIn('xczu7ev-ffvc1156-2-e', found)

    def test_part_offers_exact_parts_when_typing_xc(self):
        found = self.complete('main', 'csynth', '--part', 'xc7z')
        self.assertIn('xc7z020clg400-1', found)
        self.assertEqual(len(found), len(set(found)), 'duplicate parts offered')

    def test_kernel_offers_discovered_kernels(self):
        self.assertIn('matmul', self.complete('main', 'csim', '--kernel', ''))

    def test_flag_value_is_not_mistaken_for_the_stage(self):
        found = self.complete('main', '--kernel', 'matmul', '')
        self.assertIn('csynth', found)

    def test_stage_is_not_offered_twice(self):
        self.assertNotIn('csim', self.complete('main', 'csim', ''))

    def test_main_command_runs_the_runner(self):
        script = (f'source "{ROOT.as_posix()}/tabcompletion.sh"\nmain kernels\n')
        result = subprocess.run([BASH, '-c', script], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('matmul', result.stdout)


if __name__ == '__main__':
    unittest.main()
