from keras.layers import CuDNNGRU, LeakyReLU, Concatenate


def stacked_rnn(input_tensor, gru_cells):
    model_input = input_tensor
    x = CuDNNGRU(gru_cells, return_sequences=True)(model_input)
    x = LeakyReLU()(x)

    y = CuDNNGRU(gru_cells, return_sequences=True)(x)
    y = LeakyReLU()(y)

    z = Concatenate()([x, y])
    z = CuDNNGRU(gru_cells)(z)
    z = LeakyReLU()(z)
    return z
