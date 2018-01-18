from keras.layers import CuDNNGRU, LeakyReLU, Concatenate
from keras.constraints import Constraint
from keras import backend as K


class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}



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


def stacked_rnn_clipped(input_tensor, gru_cells, clip_value=0.01):
    model_input = input_tensor
    x = CuDNNGRU(gru_cells, return_sequences=True, W_constraint = WeightClip(clip_value))(model_input)
    x = LeakyReLU()(x)

    y = CuDNNGRU(gru_cells, return_sequences=True, W_constraint = WeightClip(clip_value))(x)
    y = LeakyReLU()(y)

    z = Concatenate()([x, y])
    z = CuDNNGRU(gru_cells, W_constraint = WeightClip(clip_value))(z)
    z = LeakyReLU()(z)
    return z
