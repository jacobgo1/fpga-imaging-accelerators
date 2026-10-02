# Tools

Host-side Python utilities that are not part of the build runner. Unlike
`main.py`, scripts here may depend on third-party packages -- install what
each script names at the top of its own file.

```text
pt_reader.py            Read PyTorch .pt checkpoints with numpy only (no torch)
extract_weights.py      Any checkpoint(s) -> weights/extracted/<model>/: every tensor
                        as C++ arrays (.hpp), as .npz, and a summary (.md); plus an
                        overview of all models. For starting a new model's kernel.
quantize_weights.py     The same, quantized: BatchNorm folded into its conv, then
                        int8 (--bits N) with a power-of-two scale per tensor
                        (--per-channel, --scale float), as Vitis AI does. Writes
                        weights/quantized/<model>/<model>_int8.hpp/.npz/.md with
                        the error (SQNR) of every tensor.
export_justoliunet.py   Checkpoint + training mu_sd.txt -> justoliunet kernel
                        (band selection, z-score, weights) + test vectors
                        (numpy). Re-run after changing either.
justoliunet_image.py    Image cube -> pixels for the board, then FPGA result
                        vs reference and labels, with PNG class maps (numpy)
hypso_dims.py   Spatial/spectral dimensions of HYPSO .nc captures
                (pip install hypso). Answers what real capture sizes look
                like before setting CONV_IC/CONV_IH/CONV_IW in a kernel.
```
