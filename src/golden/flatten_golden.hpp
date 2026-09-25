#pragma once

// Matches PyTorch's torch.flatten(x, 1) on a channels-first (N, C, L)
// tensor, applied pointwise per row of a [H][W][CH] picture (see
// conv1d_golden.hpp): each row's flattened vector is channel-major,
// length-minor -- element c*W + j is channel c, position j. The transpose
// happens in this module (by how the two inner loops are nested), not on
// the weights.
template<typename data_t, int H, int W, int CH>
void flatten_golden(data_t din[H][W][CH], data_t dout[H][CH * W]) {
    for (int i = 0; i < H; i++) {
        for (int c = 0; c < CH; c++) {
            for (int j = 0; j < W; j++) {
                dout[i][c * W + j] = din[i][j][c];
            }
        }
    }
}
