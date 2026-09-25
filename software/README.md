# Software

Host utilities, bare-metal applications and embedded Linux components, once a
board and software stack are chosen. Take AXI-Lite register offsets from the
driver headers generated inside the exported HLS IP rather than hardcoding them.

[`zcu104/`](zcu104/README.md) -- scripted, GUI-free ZCU104 bring-up (flash an
SD card, seed cloud-init for headless SSH access) and a `deploy-and-run.sh`
that rsyncs this repo to the board's Cortex-A53 and runs a `main.py` stage
there natively, for a software baseline to compare against the PL.
