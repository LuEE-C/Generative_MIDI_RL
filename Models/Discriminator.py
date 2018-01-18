from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
from Models.Stacked_RNN import stacked_rnn


class DiscriminatorNetwork(object):
    def __init__(self, cutoff, action_size, tau, lr):
        self.tau = tau
        self.lr = lr
        self.cutoff = cutoff
        self.action_size = action_size

        self.model = self.create_discriminator_network()
        self.target_model = self.create_discriminator_network()

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)


    def create_discriminator_network(self):
        state_input = Input(shape=(self.cutoff, self.action_size))

        main_network = stacked_rnn(state_input, 125)

        outputs = Dense(1, activation='sigmoid')(main_network)

        discriminator = Model(inputs=[state_input], outputs=outputs)
        discriminator.compile(optimizer=Adam(lr=self.lr), loss='binary_crossentropy')
        discriminator.summary()

        return discriminator
