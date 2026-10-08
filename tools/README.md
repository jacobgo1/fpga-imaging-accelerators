# Tools

Host-side Python, run on the PC or the lab server, never on the FPGA. Unlike
`main.py`, scripts here may depend on third-party packages: each names what it
needs at the top of its file (all need numpy; reading `.nc` captures also needs
`pip install hypso`).

## Weights: checkpoint (.pt) -> C++ for a kernel

The `.pt` checkpoints under `weights/fp32/` are the source and stay in git.
Everything these tools write (`weights/extracted/`, `weights/quantized/`) is
regenerated from them, so it is not.

```text
pt_reader.py            Reads PyTorch .pt files with numpy only (no torch), so the
                        lab server can export weights. Used by the two below.
extract_weights.py      Any checkpoint -> weights/extracted/<model>/: every tensor as
                        float C++ arrays (.hpp), .npz and a summary (.md). The starting
                        point for a new model's golden (float) kernel.
quantize_weights.py     The same, quantized: BatchNorm folded into its conv, then int8
                        with a power-of-two scale per tensor, as Vitis AI does ->
                        weights/quantized/<model>/<model>_int8.hpp/.npz/.md.
                        justounetsimple and justounetsimple_opt include this header.
```

## The model on the PC: the reference the FPGA is checked against

```text
justounetsimple_model.py  numpy model of the justounetsimple kernels: preprocessing,
                          float forward pass (the reference scores), and the bit-exact
                          integer version of justounetsimple_opt (classify_fixed). Also
                          writes src/hls/justounetsimple_opt/justounetsimple_prep.hpp
                          (the integer z-score constants): python tools/justounetsimple_model.py
patching.py               Cuts an image into 32 x 32 patches and stitches results back,
                          the same way the kernels do.
```

## Images: a HYPSO capture -> a folder for the board notebook

```text
justounetsimple_image.py  prepare: capture -> cube.npy, labels and the reference scores,
                          for the justounetsimple and justounetsimple_opt notebooks.
```
