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
preprocessing.py          what every model shares: keep raw bands 8-117 and z-score them with
                          mu_sd.txt, in float (the golden notebooks) and in integers (the
                          optimized kernels, bit for bit). Also writes the 2D kernels' integer
                          z-score constants, src/optimized/shared/zscore_int8.hpp:
                          python tools/preprocessing.py
justounetsimple_model.py  numpy model of the justounetsimple kernels: float forward pass
                          (the reference scores) and the bit-exact integer version of
                          justounetsimple_opt (classify_fixed).
justoliunet_model.py      numpy model of the justoliunet and justoliunet_bn kernels (the
                          1D per-pixel networks): the trained model with its BatchNorm, and
                          the int8 network the golden kernels run, and classify_fixed, the
                          optimized kernels (justoliunet_opt, justoliunet_bn_opt) bit for bit.
                          python tools/justoliunet_model.py compares them on the labelled
                          aegean pixels; ... formats writes the optimized kernels' integer
                          formats, src/hls/justoliunet_opt/justoliunet_opt_formats.hpp.
spectralpointwise1d_model.py  numpy model of the spectralpointwise1d kernels (1D-SpectralPointwise:
                          three 1x1 convs per pixel): the trained model with its BatchNorm, the
                          int8 network the golden kernel runs, and classify_fixed, the
                          optimized kernels (spectralpointwise1d_opt and its Zynq-7030 and
                          Zynq-7020 versions) bit for bit, on 1D-Justo-LiuNet's 12-bit input.
                          python tools/spectralpointwise1d_model.py compares them on the
                          labelled aegean pixels; ... formats writes the activations' formats,
                          src/hls/spectralpointwise1d_opt/spectralpointwise1d_opt_formats.hpp.
waveletcnn_model.py       numpy model of the waveletcnn kernels: float forward pass (1D db4
                          DWT + 1x1 projection + 3x3 conv_block + 1x1 seg_head), and the
                          bit-exact integer version of waveletcnn_opt (classify_fixed).
specpw_unet_model.py      numpy model of SpecPW-UNet (sp_unet_small/medium/large): the trained
                          model in float (the reference of src/golden/specpw_unet_golden.hpp),
                          the int8-weight network the golden kernels run (forward_patch,
                          classify), and classify_fixed, the integer arithmetic of an optimized
                          kernel bit for bit (also src/golden/specpw_unet_fixed_golden.hpp).
                          python tools/specpw_unet_model.py prints each size's accuracy on the
                          labelled aegean pixels.
models.py                 every model in one table (its patch size, its scores' format), which
                          main.py packs with the board's notebook as kernel.json, and the
                          PC's answers for a cube (golden_scores, fixed_scores), which
                          tests/test_notebooks.py checks the notebooks with. The board itself
                          is checked against the image's labels only.
patching.py               Cuts an image into 32 x 32 patches and stitches results back,
                          the same way the kernels do.
```

The labelled aegean pixels are in `aegean/` at the top of the repo (`pixels.txt`, `labels.npy`,
`reference_logits.npy`). It is not in git: copy it in when needed. Without it, these scripts and
the tests that use it (test_justoliunet, test_justoliunet_opt, test_spectralpointwise1d,
test_specpw_unet) skip.

## Images: a HYPSO capture -> a folder for the board notebook

```text
image.py                  prepare: capture -> cube.npy (uint16 raw counts), labels.npy and
                          meta.json (the picture's bands).
```

One image folder serves every kernel: prepare it and zip it.

```bash
python3 tools/image.py prepare CAPTURE-l1a.nc --labels CAPTURE-l1a_labels.npy --out aegean_unet
source fpga.sh; fpga_zip aegean_unet
```
