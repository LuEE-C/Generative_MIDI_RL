from keras.layers import Input, Dense, Conv1D, LeakyReLU, Dropout, MaxPooling1D, GlobalMaxPooling1D, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from Models.Stacked_RNN import stacked_rnn
from keras.layers.merge import _Merge
from functools import partial

GRADIENT_PENALTY_WEIGHT = 10
BATCH_SIZE = 256
# GP taken from https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(K.sum(y_pred), averaged_samples)
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return gradient_penalty


class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


class DiscriminatorNetwork(object):
    def __init__(self, cutoff, action_size, lr, tau):
        self.lr = lr
        self.tau = tau
        self.cutoff = cutoff
        self.action_size = action_size

        self.model, self.training_model = self.create_discriminator_network()
        # self.target_model, self.target_training_model = self.create_discriminator_network()

    # def target_train(self):
    #     discriminator_weights = self.model.get_weights()
    #     discriminator_target_weights = self.target_model.get_weights()
    #     for i in range(len(discriminator_weights)):
    #         discriminator_target_weights[i] = self.tau * discriminator_weights[i] + (1 - self.tau)* discriminator_target_weights[i]
    #     self.target_model.set_weights(discriminator_target_weights)


    def create_discriminator_network(self):

        # Defining discriminator
        # Second order gradient not supported in tensorflow, will have to switch to CNTK or even Theano
        # For now Conv1D based
        state_input = Input(shape=(self.cutoff, self.action_size))

        main_network = Conv1D(128, 3, padding='same')(state_input)
        main_network = LeakyReLU()(main_network)
        main_network = MaxPooling1D()(main_network)
        main_network = Dropout(0.5)(main_network)

        main_network = Conv1D(256, 3, padding='same')(main_network)
        main_network = LeakyReLU()(main_network)
        main_network = MaxPooling1D()(main_network)
        main_network = Dropout(0.5)(main_network)

        main_network = Conv1D(512, 3, padding='same')(main_network)
        main_network = LeakyReLU()(main_network)
        main_network = GlobalMaxPooling1D()(main_network)
        main_network = Dropout(0.5)(main_network)

        main_network = Dense(256)(main_network)
        main_network = LeakyReLU()(main_network)
        main_network = Dropout(0.5)(main_network)

        outputs = Dense(1)(main_network)

        discriminator = Model(inputs=[state_input], outputs=[outputs])

        # All of this is for the GP loss function
        real_input = Input(shape=(self.cutoff, self.action_size))
        fake_input = Input(shape=(self.cutoff, self.action_size))

        discriminator_output_from_generator = discriminator(fake_input)
        discriminator_output_from_real_samples = discriminator(real_input)

        averaged_samples = RandomWeightedAverage()([real_input, fake_input])
        averaged_samples_out = discriminator(averaged_samples)

        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=averaged_samples,
                                  gradient_penalty_weight=10)

        partial_gp_loss.__name__ = 'gradient_penalty'

        discriminator_model = Model(inputs=[real_input, fake_input],
                                    outputs=[discriminator_output_from_real_samples,
                                             discriminator_output_from_generator,
                                             averaged_samples_out])
        discriminator_model.compile(optimizer=Adam(self.lr, beta_1=0.5, beta_2=0.9),
                                    loss=[wasserstein_loss,
                                          wasserstein_loss,
                                          partial_gp_loss])
        discriminator.summary()

        return discriminator, discriminator_model
