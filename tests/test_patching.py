"""tools/patching.py: cutting an image into kernel patches and stitching results back."""
from pathlib import Path
import sys
import unittest

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))


@unittest.skipIf(np is None, 'patching needs numpy')
class PatchingTests(unittest.TestCase):
    def setUp(self):
        import patching
        self.p = patching

    def image(self, h, w, c=4):
        return np.arange(h * w * c, dtype=np.float32).reshape(h, w, c)

    def test_stitching_untouched_patches_returns_the_image(self):
        # 598 x 1092 is the real capture size: not a multiple of 32.
        for h, w in [(32, 32), (64, 96), (598, 1092), (40, 33), (5, 7)]:
            for halo in (0, 4, 8):
                img = self.image(h, w)
                patches, tiles = self.p.cut(img, halo=halo)
                self.assertEqual(patches.shape[1:3], (32, 32))
                out = self.p.stitch(patches, tiles, img.shape, halo=halo)
                np.testing.assert_array_equal(out, img, err_msg=f'{h}x{w} halo {halo}')

    def test_patch_count(self):
        patches, _ = self.p.cut(self.image(598, 1092), halo=0)
        self.assertEqual(len(patches), 19 * 35)          # ceil(598/32) * ceil(1092/32)
        patches, _ = self.p.cut(self.image(598, 1092), halo=4)
        self.assertEqual(len(patches), 25 * 46)          # step 24

    def test_halo_gives_real_neighbours_not_zeros(self):
        img = self.image(64, 64, 1)
        patches, tiles = self.p.cut(img, halo=4)
        # The second patch in the first row starts 24 columns in; its first 4
        # columns are image columns 20..23, which belong to the first patch.
        second = patches[1]
        np.testing.assert_array_equal(second[4:, :4, 0], img[0:28, 20:24, 0])

    def test_results_can_have_a_different_channel_count(self):
        img = self.image(50, 70, 110)
        patches, tiles = self.p.cut(img)
        scores = patches[..., :3]                         # stands in for the kernel's 3 scores
        out = self.p.stitch(scores, tiles, img.shape)
        self.assertEqual(out.shape, (50, 70, 3))
        np.testing.assert_array_equal(out, img[..., :3])

    def test_bad_halo_is_rejected(self):
        with self.assertRaises(ValueError):
            self.p.cut(self.image(64, 64), halo=16)


if __name__ == '__main__':
    unittest.main()
