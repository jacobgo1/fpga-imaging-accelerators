"""Two numbers the ZedBoard overlay needs, from a build's system.hwh:

    python3 hwh_values.py system.hwh   ->   40000000 78125000

the kernel's register address on GP0 (hex, no 0x) and FCLK_CLK0 in Hz, as the
PLL really makes it (rounded down, so the clock never ends up faster than the
kernel was built for).
"""
import re
import sys

text = open(sys.argv[1], encoding='utf-8', errors='replace').read()
memrange = re.search(r'<MEMRANGE\b[^>]*\bINSTANCE="kernel"[^>]*>', text)
base = memrange and re.search(r'\bBASEVALUE="0x([0-9A-Fa-f]+)"', memrange.group(0))
clock = re.search(r'NAME="PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ"\s+VALUE="([0-9.]+)"', text)
if not base or not clock:
    sys.exit(f'{sys.argv[1]}: no {"address of the kernel" if not base else "FCLK_CLK0 frequency"} in it')
print(f'{int(base.group(1), 16):x} {int(float(clock.group(1)) * 1e6)}')
