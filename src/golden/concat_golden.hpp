#pragma once

// torch.cat([first, second], dim=1): each pixel's channels of first, then of second.
// The order matters: SpecPW-UNet's accuracy drops from 95 % to 30-40 % the other way round.
template<typename data_t, int HEIGHT, int WIDTH, int FIRST_CHANNELS, int SECOND_CHANNELS>
void concat_golden(const data_t first[HEIGHT][WIDTH][FIRST_CHANNELS],
                   const data_t second[HEIGHT][WIDTH][SECOND_CHANNELS],
                   data_t dout[HEIGHT][WIDTH][FIRST_CHANNELS + SECOND_CHANNELS]) {
    for (int row = 0; row < HEIGHT; row++)
        for (int col = 0; col < WIDTH; col++) {
            for (int ch = 0; ch < FIRST_CHANNELS; ch++)
                dout[row][col][ch] = first[row][col][ch];
            for (int ch = 0; ch < SECOND_CHANNELS; ch++)
                dout[row][col][FIRST_CHANNELS + ch] = second[row][col][ch];
        }
}
