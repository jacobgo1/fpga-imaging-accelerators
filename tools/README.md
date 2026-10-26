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
justoliunet_model.py      numpy model of the justoliunet and justoliunet_bn kernels (the
                          1D per-pixel networks): the trained model with its BatchNorm, and
                          the int8 network the golden kernels run, and classify_fixed, the
                          optimized kernels (justoliunet_opt, justoliunet_bn_opt) bit for bit.
                          python tools/justoliunet_model.py compares them on the labelled
                          aegean pixels; ... formats writes the optimized kernels' integer
                          formats, src/hls/justoliunet_opt/justoliunet_opt_formats.hpp;
                          ... reference IMAGE writes IMAGE/reference_<model>.npy, what every
                          optimized kernel writes for IMAGE/cube.npy (the ZedBoard's classify
                          and the justoliunet notebooks compare with it; the board does not
                          run the model itself).
waveletcnn_model.py       numpy model of the waveletcnn kernels: preprocessing, float
                          forward pass (1D db4 DWT + 1x1 projection + 3x3 conv_block +
                          1x1 seg_head), and the bit-exact integer version of waveletcnn_opt
                          (classify_fixed). Also writes
                          src/hls/waveletcnn_opt/waveletcnn_prep.hpp:
                          python tools/waveletcnn_model.py
specpw_unet_model.py      numpy model of SpecPW-UNet (sp_unet_small/medium/large): the trained
                          model in float (the reference of src/golden/specpw_unet_golden.hpp),
                          the int8-weight network the golden kernels run (forward_patch,
                          classify), and classify_fixed, the integer arithmetic of an optimized
                          kernel bit for bit (also src/golden/specpw_unet_fixed_golden.hpp).
                          python tools/specpw_unet_model.py prints each size's accuracy on the
                          labelled aegean pixels; ... reference IMAGE writes
                          IMAGE/reference_sp_unet_SIZE.npy, what each golden kernel answers
                          for IMAGE/cube.npy (the notebook compares the board with it);
                          ... fixed IMAGE writes IMAGE/reference_fixed_sp_unet_SIZE.npy, what
                          each optimized kernel sp_unet_SIZE_opt writes, bit for bit.
patching.py               Cuts an image into 32 x 32 patches and stitches results back,
                          the same way the kernels do.
```

## Images: a HYPSO capture -> a folder for the board notebook

```text
justounetsimple_image.py  prepare: capture -> cube.npy, labels and the reference scores,
                          for the justounetsimple and justounetsimple_opt notebooks.
waveletcnn_image.py       prepare: capture -> cube.npy, labels and the reference scores,
                          for the waveletcnn and waveletcnn_opt notebooks.
```

Every notebook checks the FPGA's classes against `labels.npy`, and against the PC's answer
when the image folder has it; nothing big is computed on the board's ARM (the models there
ran out of memory on whole captures). For the 1D networks' answers too, before zipping:

```bash
python3 tools/justounetsimple_image.py prepare CAPTURE-l1a.nc --labels CAPTURE-l1a_labels.npy --out aegean_unet
python3 tools/justoliunet_model.py reference aegean_unet
python3 tools/specpw_unet_model.py reference aegean_unet
python3 tools/specpw_unet_model.py fixed aegean_unet
source fpga.sh; fpga_zip aegean_unet
```
