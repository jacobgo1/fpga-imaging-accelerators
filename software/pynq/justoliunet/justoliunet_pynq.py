"""Drive the justoliunet kernel from a PYNQ notebook on the ZCU104.

    from justoliunet_pynq import JustoLiuNet
    jl = JustoLiuNet()               # loads justoliunet.bit + .hwh from this folder
    jl.selftest()                    # 15 known pixels against their expected results
    jl.classify(spectrum)            # 120 raw L1a values -> 3 scores (cloud, land, sea)
    jl.classify_folder('aegean')     # a folder (or .zip) made by justoliunet_image.py prepare

The kernel's registers (from xjustoliunet_hw.h, next to this file):
AP_CTRL (bit 0 start, bit 1 done, bit 2 idle), `spectrum` (120 float32
words in) and `logits` (3 float32 words out). One pixel per start.
"""
import json
from pathlib import Path
import re
import time
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
CLASS_NAMES = ('cloud', 'land', 'sea')
TOLERANCE = (1e-4, 1e-4)  # absolute, relative: float32 hardware vs float64 reference
AP_START, AP_DONE, AP_IDLE = 0x1, 0x2, 0x4


class RegisterMap:
    """Offsets of the kernel's AXI-Lite registers, from the HLS driver header."""

    def __init__(self, folder=HERE):
        headers = sorted(Path(folder).glob('x*_hw.h'))
        if len(headers) != 1:
            raise FileNotFoundError(f'expected one x*_hw.h register map in {folder}, found {len(headers)}')
        text = headers[0].read_text()
        self.ctrl = int(re.search(r'#define\s+X\w*?_ADDR_AP_CTRL\s+(0x[0-9a-fA-F]+)', text).group(1), 16)
        self.arrays = {}
        for name, base in re.findall(r'#define\s+X\w*?_ADDR_(\w+)_BASE\s+(0x[0-9a-fA-F]+)', text):
            width = int(re.search(rf'#define\s+X\w*?_WIDTH_{name}\s+(\d+)', text).group(1))
            depth = int(re.search(rf'#define\s+X\w*?_DEPTH_{name}\s+(\d+)', text).group(1))
            if width != 32:
                raise ValueError(f'{name} has {width}-bit elements; this driver handles 32-bit arrays')
            self.arrays[name.lower()] = (int(base, 16), depth)


def kernel_ip(overlay):
    """The kernel in the loaded design: the block named 'kernel', or the HLS justoliunet IP."""
    for name, info in overlay.ip_dict.items():
        if name == 'kernel' or 'justoliunet' in str(info.get('type', '')):
            return getattr(overlay, name)
    raise LookupError(f'no justoliunet kernel in this design; IP blocks: {list(overlay.ip_dict)}')


def load_vectors(path=HERE / 'justoliunet_vectors.txt'):
    """Test pixels: (raw spectra, expected logits)."""
    inputs, expected = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        x, y = line.split('|')
        inputs.append(np.array(x.split(), dtype=np.float32))
        expected.append(np.array(y.split(), dtype=np.float64))
    return np.array(inputs), np.array(expected)


def open_image(path):
    """A prepared image folder; a .zip of one is unpacked next to it first."""
    path = Path(path)
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as archive:
            archive.extractall(path.parent)
            tops = {Path(n).parts[0] for n in archive.namelist()}
        path = path.parent / (tops.pop() if len(tops) == 1 else path.stem)
    if not (path / 'pixels.txt').is_file():
        raise FileNotFoundError(f'{path} has no pixels.txt; make it with justoliunet_image.py prepare')
    return path


class JustoLiuNet:
    def __init__(self, folder=HERE, overlay=None):
        folder = Path(folder)
        if overlay is None:
            from pynq import Overlay  # also sets the FPGA clock and PS port widths from the .hwh
            overlay = Overlay(str(folder / 'justoliunet.bit'))
        self.overlay = overlay
        self.mmio = kernel_ip(overlay).mmio
        self.regs = RegisterMap(folder)
        self.spectrum_offset, self.bands = self.regs.arrays['spectrum']
        self.logits_offset, self.classes = self.regs.arrays['logits']

    def status(self):
        v = self.mmio.read(self.regs.ctrl)
        return {'start': bool(v & AP_START), 'done': bool(v & AP_DONE), 'idle': bool(v & AP_IDLE)}

    def classify(self, spectrum, timeout=2.0):
        """One pixel on the FPGA: raw spectrum (120 values) -> 3 logits."""
        words = np.asarray(spectrum, dtype=np.float32).view(np.uint32)
        if len(words) != self.bands:
            raise ValueError(f'the kernel takes {self.bands} bands, got {len(words)}')
        for i, word in enumerate(words):  # one 32-bit write each, as AXI-Lite needs
            self.mmio.write(self.spectrum_offset + 4 * i, int(word))
        self.mmio.write(self.regs.ctrl, AP_START)
        deadline = time.monotonic() + timeout
        while not self.mmio.read(self.regs.ctrl) & AP_DONE:  # done clears when read
            if time.monotonic() > deadline:
                raise TimeoutError(f'kernel did not finish within {timeout} s: {self.status()}')
        out = [self.mmio.read(self.logits_offset + 4 * i) for i in range(self.classes)]
        return np.array(out, dtype=np.uint32).view(np.float32).astype(np.float64)

    @staticmethod
    def class_name(logits):
        c = int(np.argmax(logits))
        return f'{c} {CLASS_NAMES[c]}' if c < len(CLASS_NAMES) else str(c)

    def classify_pixels(self, pixels, report_every=1024):
        pixels = np.asarray(pixels, dtype=np.float32).reshape(-1, self.bands)
        logits = np.empty((len(pixels), self.classes))
        start = time.monotonic()
        for n, spectrum in enumerate(pixels, start=1):
            logits[n - 1] = self.classify(spectrum)
            if n % report_every == 0 or n == len(pixels):
                elapsed = time.monotonic() - start
                print(f'  {n}/{len(pixels)} pixels, {elapsed:.1f} s, {1000 * elapsed / n:.2f} ms/pixel, '
                      f'~{elapsed / n * (len(pixels) - n):.0f} s left', flush=True)
        return logits

    def selftest(self, vectors=HERE / 'justoliunet_vectors.txt'):
        inputs, expected = load_vectors(vectors)
        failed = 0
        for i, (x, want) in enumerate(zip(inputs, expected)):
            got = self.classify(x)
            atol, rtol = TOLERANCE
            ok = bool(np.all(np.abs(got - want) <= atol + rtol * np.abs(want)) and got.argmax() == want.argmax())
            failed += not ok
            print(f'{"PASS" if ok else "FAIL"} pixel {i:2d}  {self.class_name(got)}'
                  + ('' if ok else f'   got {np.round(got, 5)}, expected {np.round(want, 5)}'))
        print(f'{failed} of {len(inputs)} pixels FAILED' if failed
              else f'PASS: all {len(inputs)} pixels match the reference')
        return failed == 0

    def classify_folder(self, path):
        """Classify a prepared image; writes fpga_logits.txt for justoliunet_image.py compare."""
        folder = open_image(path)
        pixels = np.loadtxt(folder / 'pixels.txt', dtype=np.float32, comments='#', ndmin=2)
        shape = json.loads((folder / 'meta.json').read_text())['shape']
        print(f'{folder}: {len(pixels)} pixels ({shape[0]} x {shape[1]})')
        logits = self.classify_pixels(pixels)
        np.savetxt(folder / 'fpga_logits.txt', logits, fmt='%.9g')
        return folder, logits
