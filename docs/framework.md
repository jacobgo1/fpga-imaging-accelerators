# How it works: from C++ to a result in the notebook

There are two halves. **On the lab server**, one command turns a kernel's C++
into a bitstream and a zip for the board. **On the board**, a Jupyter notebook
(PYNQ) loads that bitstream and drives the kernel. The example used here is
`justounetsimple_opt`, but every kernel goes the same way.

## 1. Building: `fpga_build justounetsimple_opt`

That runs `python3 main.py bitstream --kernel justounetsimple_opt --skip-cosim`,
which does five things in order:

| # | Who | What | Result |
| --- | --- | --- | --- |
| 1 | `main.py` | Finds `src/hls/K/*.cpp` and `tb/K/*.cpp`, makes `build/K/DATE-bitstream/` and writes `settings.tcl` there (paths, part, clock) | a run folder |
| 2 | Vitis HLS runs [`scripts/hls.tcl`](../scripts/hls.tcl) | C simulation (your testbench) → C synthesis (C++ → Verilog) → `export_design` | an **IP block** in `hls/solution/impl/ip/` |
| 3 | Vivado runs [`scripts/vivado.tcl`](../scripts/vivado.tcl) | Empty project, the IP put in its catalog, then [`boards/zcu104/system.tcl`](../boards/zcu104/system.tcl) | a **block design**: the hardware around the IP (section 2) |
| 4 | Vivado, still `vivado.tcl` | Synthesis → place and route → timing check → `.bit` | the bitstream in `vivado/system.runs/impl_1/` |
| 5 | `main.py` | Takes the `.bit`, the `.hwh` (block design description) and `xK_hw.h` (register offsets) from the run folder, names the first two after the kernel, adds `software/pynq/K/` (notebook) and the files in its `include.txt`, and zips it | `K_pynq.zip` in the run folder |

Then upload the zip to Jupyter on the board, unzip it there, and run the notebook.

Everything stays in that one run folder. Where to look afterwards:

```text
build/K/latest/hls/solution/syn/report/K_csynth.rpt   HLS estimate: latency, resources
build/K/latest/reports/impl/                          real utilization and timing
build/K/latest/vivado/system.xpr                      the Vivado project: open it to see the design
build/K/latest/K_pynq.zip                             what goes to the board
build/K/latest/run.json                               what was built: git commit, part, clock, result
```

`main.py bitstream` without `--skip-cosim` also runs C/RTL co-simulation in
step 2. It's slow, so `fpga_build` skips it. Run `main.py cosim --kernel K`
separately when the RTL needs checking.

## 2. The hardware around the kernel

`boards/zcu104/system.tcl` creates every block and wire of this design, with
nothing added automatically:

```text
 PS ("ps": 4 x ARM, DDR controller)
   pl_clk0  ───────────────────────────────┬──► every clock pin below (100 MHz)
   pl_resetn0 ──► rst (proc_sys_reset) ────┴──► every reset pin below
   M_AXI_HPM0_FPD ──► ctrl (SmartConnect) ──► kernel/s_axi_control   registers
   S_AXI_HP0_FPD  ◄── data (SmartConnect) ◄── kernel/m_axi_gmem0     reads input from DDR
                                          ◄── kernel/m_axi_gmem1     writes output to DDR
   pl_ps_irq0     ◄── kernel/interrupt                               (not used)
```

The kernel's ports come from the `INTERFACE` pragmas on its top function:

| Pragma in the C++ | Port on the IP | Wired to |
| --- | --- | --- |
| `s_axilite port=... bundle=control` (and `port=return`) | `s_axi_control`: start/done bits, scalar arguments, buffer addresses | PS `M_AXI_HPM0_FPD`, through `ctrl` |
| `m_axi port=din bundle=gmem0` | `m_axi_gmem0`: the kernel reads/writes DDR itself | PS `S_AXI_HP0_FPD`, through `data` |
| (always) | `ap_clk`, `ap_rst_n`, `interrupt` | `pl_clk0`, `rst`, `pl_ps_irq0` |

A kernel with no `m_axi` ports (e.g. `matmul`) gets only the control path.

### Changing it

- **Looking:** after a build, open `build/K/latest/vivado/system.xpr` in Vivado
  (on the lab server) and open the block design. That's exactly what was built.
- **Changing:** edit `system.tcl`. Each block is a `create_bd_cell`, each wire
  a `connect_bd_net` (single pins: clock, reset) or `connect_bd_intf_net`
  (whole AXI buses). Examples:
  - *A different clock:* `clock_ns` in `config/project.json` sets both the HLS
    target and `pl_clk0`.
  - *An AXI-Stream kernel with a DMA:* add an `axi_dma` cell, connect its
    `M_AXIS_MM2S` to the kernel's input stream and the kernel's output stream to
    `S_AXIS_S2MM`, its `S_AXI_LITE` to a second `ctrl` master (`NUM_MI 2`), and
    its memory ports to `data`.
  - *Trying it by hand first:* change the design in the GUI in the opened
    project, run `write_bd_tcl -force try.tcl` in the Tcl console, and copy the
    new `create_bd_cell` / `connect_bd_*` lines from `try.tcl` into
    `system.tcl`.
- **Rules:** the block named `kernel` must stay `kernel` (the notebook uses
  `overlay.kernel`), and the rest of `vivado.tcl` expects the script to leave
  a validated design with `system_wrapper` as the top.

## 3. Running: the notebook on the board

The board runs PYNQ (Linux), with Jupyter at `http://BOARD:9090`. After
`!unzip -o K_pynq.zip`, the notebook does this:

1. `Overlay('K.bit')`: PYNQ programs the FPGA. From `K.hwh` (same name, same
   folder) it sets `pl_clk0` and finds the block called `kernel`.
2. `allocate(...)` reserves buffers in DDR that the kernel can reach through
   HP0. Each has a `physical_address`.
3. Registers are written over `ctrl`: the buffer addresses into `din`/`dout`,
   and arguments such as `n`. The offsets come from `xK_hw.h`, which HLS
   generated.
4. `AP_CTRL = 1` starts the kernel. It reads its input from DDR through
   `m_axi_gmem0` → `data` → HP0 and writes the result back the same way.
5. The notebook polls `AP_CTRL` bit 1 (done), calls `invalidate()` so the CPU
   doesn't read stale cache, and reads the result buffer.

Before step 4, `flush()` pushes the input from the CPU's cache into DDR, where
the kernel can see it.

## 4. Commands

```bash
source fpga.sh       # once per terminal on the lab server
fpga_help            # every command, and the main.py line it runs
```

| Goal | Command |
| --- | --- |
| Tools on PATH | `fpga_env` |
| Test on the PC/server, no FPGA | `fpga_test K` |
| Bitstream + board zip | `fpga_build K` |
| List builds | `fpga_runs K` |
| Re-make the zip of an old build (e.g. after editing the notebook) | `fpga_pynq K [RUN]` |
| Zip an image folder for upload | `fpga_zip DIR` |

Other steps are plain Python scripts in `tools/` (weights, quantization,
preparing images), described in [tools/README.md](../tools/README.md).

## 5. Adding a kernel that runs on the board

1. `src/hls/K/K.cpp` with the top function `K` and its `INTERFACE` pragmas
   (copy them from `justounetsimple_opt.cpp`), plus `tb/K/K_tb.cpp`.
2. `fpga_test K` until it passes.
3. `software/pynq/K/K.ipynb` (copy `justounetsimple_opt.ipynb` and change the
   register names and buffer shapes), plus an `include.txt` if it needs repo files.
4. `fpga_build K`, upload `K_pynq.zip`, and run the notebook.

The first accelerator, `justoliunet`, matched the PC reference on every pixel
of a real capture (`aegean/overview.png`). The JTAG/xsdb scripts used for that
run were removed. They're under the git tag `pre-simplify`.
