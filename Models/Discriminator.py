from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from Models.Stacked_RNN import stacked_rnn
from keras.layers.merge import _Merge

# GP taken from https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(K.sum(y_pred), averaged_samples)
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return gradient_penalty



class DiscriminatorNetwork(object):
    def __init__(self, cutoff, action_size, tau, lr, batch_size):
        self.tau = tau
        self.lr = lr
        self.cutoff = cutoff
        self.action_size = action_size

        self.model = self.create_discriminator_network()

        class RandomWeightedAverage(_Merge):

            def _merge_function(self, inputs):
                weights = K.random_uniform((batch_size, 1, 1, 1))
                return (weights * inputs[0]) + ((1 - weights) * inputs[1])

    def create_discriminator_network(self):
        state_input = Input(shape=(self.cutoff, self.action_size))

        real_input = Input(shape=(self.cutoff, self.action_size))
        fake_input = Input(shape=(self.cutoff, self.action_size))
        average_samples =

        main_network = stacked_rnn(state_input, 100)

        outputs = Dense(1)(main_network)

        discriminator = Model(inputs=[state_input], outputs=outputs)
        discriminator.compile(optimizer=Adam(lr=self.lr), loss=wasserstein_loss)
        discriminator.summary()

        return discriminator
