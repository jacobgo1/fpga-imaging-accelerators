# Tools

Host-side Python utilities that are not part of the build runner. Unlike
`main.py`, scripts here may depend on third-party packages -- install what
each script names at the top of its own file.

```text
hypso_dims.py   Spatial/spectral dimensions of HYPSO .nc captures
                (pip install hypso). Answers what real capture sizes look
                like before setting CONV_IC/CONV_IH/CONV_IW in a kernel.
```
