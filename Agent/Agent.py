import os

import numba as nb
import numpy as np
from math import log

import keras.backend as K
from Models.Actor import ActorNetwork
from Models.Critic import CriticNetwork
from Models.Discriminator import DiscriminatorNetwork

import tensorflow as tf

from Environnement.Environnement import Environnement
from PriorityExperienceReplay.PriorityExperienceReplay import Experience

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Agent:
    def __init__(self, cutoff=15, from_save=False, gamma=.99, batch_size=64, min_history=64000, lr=10e-4,
                 discriminator_loss_limits=0.2, n_steps=5, tau=0.001):

        self.cutoff = cutoff
        self.environnement = Environnement(cutoff=cutoff)

        self.batch_size = batch_size

        self.n_steps = n_steps

        # self.labels = np.array([1] * self.batch_size + [0] * self.batch_size)
        self.labels = np.array([-1] * self.batch_size + [1] * self.batch_size)
        self.gamma = gamma

        self.tau = tau

        self.min_history = min_history
        self.lr = lr

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        K.set_learning_phase(1)
        K.set_session(self.sess)

        self.discriminator_loss_limit = discriminator_loss_limits

        self.actor = ActorNetwork(self.sess, self.cutoff, 3, self.tau, self.lr)
        self.critic = CriticNetwork(self.sess, self.cutoff, 3, self.tau, self.lr)

        self.discriminator = DiscriminatorNetwork(self.cutoff, 3, self.tau, self.lr, self.batch_size)
        self.discriminator_training_batch, self.discriminator_total_batch = 0, 0

        self.memory = Experience(memory_size=1000000, batch_size=self.batch_size, alpha=0.5)

        self.dataset_epoch = 0

    # Main loop
    def train(self, epoch):

        e, total_frames = 0, 0
        while e <= epoch:

            while self.memory.tree.size < self.min_history:
                self.add_data_to_memory()

            while True:

                if total_frames % 4 == 0:

                    self.add_data_to_memory()

                self.train_discriminator()
                self.train_on_replay()

                total_frames += 1

                if total_frames % 2000 == 0:
                    print('Epoch :', e,
                          '\tDataset Epoch :', self.dataset_epoch,
                          end='\t'
                          )
                    # self.discriminator_total_batch, self.discriminator_training_batch = 0, 0
                    eval_seed, eval_loss = self.make_big_seed(3)
                    print('Eval loss :', '%.4f' % eval_loss, end='\t')
                    self.environnement.make_midi(eval_seed, str(e) + '.mid')
                    self.actor.model.save('saved_models/actor_' + str(e))
                    self.critic.model.save('saved_models/critic_' + str(e))
                    self.discriminator.model.save('saved_models/discriminator_' + str(e))
                    e += 1

    # @nb.jit
    def train_discriminator(self, evaluate=False):
        fake_batch = self.get_fake_batch()
        real_batch, done = self.environnement.query_state(self.batch_size)
        if done is True:
            self.dataset_epoch += 1
        batch = np.vstack((real_batch, fake_batch))
        if evaluate is True:
            # self.discriminator_total_batch += 1
            return self.discriminator.model.evaluate([batch], [self.labels], verbose=0)
        # self.discriminator_total_batch += 1
        # self.discriminator_training_batch += 1
        for l in self.discriminator.model.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -1, 1) for w in weights]
            l.set_weights(weights)

        return self.discriminator.model.train_on_batch([batch], [self.labels])

    def get_fake_batch(self):
        seed = self.get_seed()
        states = np.zeros((self.batch_size + 1, self.cutoff, 3))
        states[0] = seed
        for i in range(self.batch_size):
            action = self.actor.target_model.predict(states[i:i+1])
            states[i+1, :-1] = states[i, 1:]
            states[i+1, -1] = action
        return states[:-1]


    def make_training_data(self):
        seed = self.get_seed()
        states = np.zeros((self.batch_size + 1, self.cutoff, 3))
        actions = np.zeros((self.batch_size + 1, 3))
        states[0] = seed
        for i in range(self.batch_size):
            actions[i] = np.clip(self.actor.model.predict(states[i:i + 1]) + np.random.normal(loc=0, scale=1, size=(3,)), -1, 1)
            states[i+1, :-1] = states[i, 1:]
            states[i+1, -1] = actions[i]

        return states[:self.batch_size], actions[:self.batch_size], states[1:self.batch_size + 1]



    def add_data_to_memory(self):
        states, actions, states_prime = self.make_training_data()
        for i in range(self.batch_size):
            self.memory.add((states[i], actions[i], states_prime[i]), 10)

    @nb.jit
    def calc_rewards(self, states, states_primes):
        rewards = np.zeros((self.batch_size, 1))
        for i in range(self.batch_size):
            # val = self.discriminator.target_model.evaluate(states[i:i + 1], np.ones((1, 1)), verbose=0)
            val = self.discriminator.model.evaluate(states[i:i + 1], np.ones((1, 1)), verbose=0)
            if val < 0:
                val = min(-log(abs(val)), 0)
            else:
                val = max(log(val), 0)

            rewards[i] = val
            rewards[i,0] += self.critic.target_model.predict([states_primes[i:i+1],
                                    self.actor.target_model.predict(states_primes[i:i+1])]) * self.gamma

        return rewards

    def get_seed(self, seed=False):
        if seed is False:
            seed = np.clip(np.random.normal(0, 1, (1, self.cutoff, 3)), -1, 1)
        for _ in range(self.cutoff):
            prediction = self.actor.target_model.predict(seed)
            seed[:, :-1] = seed[:, 1:]
            seed[:, -1] = prediction
        return seed

    def make_big_seed(self, times=10):
        seed_list = [self.get_seed()]
        for _ in range(times):
            seed_list.append(self.get_seed(seed_list[-1]))
        # loss = np.mean([self.discriminator.target_model.evaluate(seed_list[i], np.ones((1, 1)), verbose=0) for i in
        #                     range(times)])
        loss = np.mean([self.discriminator.model.evaluate(seed_list[i], np.ones((1, 1)), verbose=0) for i in
                            range(times)])
        seed = np.concatenate(seed_list, axis=1)
        return seed, np.mean(loss)

    def make_dataset(self):
        data, weights, indices = self.memory.select(0.6)
        states, actions, states_prime = [], [], []
        for i in range(self.batch_size):
            states.append(data[i][0])
            actions.append(data[i][1])
            states_prime.append(data[i][2])
        states = np.array(states)
        actions = np.array(actions)
        states_prime = np.array(states_prime)
        return states, actions, states_prime, indices

    def train_on_replay(self):
        states, actions, states_prime, indices = self.make_dataset()
        rewards = self.calc_rewards(states, states_prime)
        loss = self.critic.model.train_on_batch([states, actions], rewards)

        a_for_grad = self.actor.model.predict(states)
        grads = self.critic.gradients(states, a_for_grad)

        self.actor.train(states, grads)
        self.actor.target_train()
        self.critic.target_train()
        self.memory.priority_update(indices, [loss for _ in range(len(indices))])



if __name__ == '__main__':
    agent = Agent(cutoff=30, batch_size=128)
    agent.train(epoch=5000)
