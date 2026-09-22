#pragma once

template<typename data_t, int Hin, int Win, int IM_CH, int FACTOR>
void nearest_neighbor_golden(data_t din[Hin][Win][IM_CH], data_t dout[Hin*FACTOR][Win*FACTOR][IM_CH]) {
    for (int i = 0; i < Hin; i++) {
        for (int j = 0; j < Win; j++) {
            for (int k = 0; k < IM_CH; k++) {
                for (int i_copy = 0; i_copy < FACTOR; i_copy++) {
                    for (int j_copy = 0; j_copy < FACTOR; j_copy++) {
                        dout[i*FACTOR + i_copy][j*FACTOR + j_copy][k] = din[i][j][k];
                    }
                }
            }
        }
    }
}
