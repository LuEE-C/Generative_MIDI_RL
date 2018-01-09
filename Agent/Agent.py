import os

import numba as nb
import numpy as np
import math
from sklearn.metrics import accuracy_score

import keras.backend as K
from keras.optimizers import Adam
from keras.layers import Input, Dense
from keras.models import Model
from Actor import ActorNetwork
from Critic import CriticNetwork
from Stacked_RNN import stacked_rnn

import tensorflow as tf

from Environnement.Environnement import Environnement
from PriorityExperienceReplay.PriorityExperienceReplay import Experience

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Agent:
    def __init__(self, cutoff=15, from_save=False, gamma=.99, batch_size=64, min_history=64000, lr=10e-4,
                 atoms=51, discriminator_loss_limits=0.2, n_steps=5, tau=0.001):

        self.cutoff = cutoff
        self.environnement = Environnement(cutoff=cutoff)

        self.batch_size = batch_size

        self.n_steps = n_steps

        self.labels = np.array([1] * self.batch_size + [0] * self.batch_size)
        self.gammas = np.array([gamma ** (i + 1) for i in range(self.n_steps + 1)]).astype(np.float32)

        self.atoms = atoms
        self.v_max = 2
        self.v_min = 0
        self.delta_z = (self.v_max - self.v_min) / float(self.atoms - 1)
        self.z_steps = np.array([self.v_min + i * self.delta_z for i in range(self.atoms)]).astype(np.float32)

        self.tau = tau

        self.min_history = min_history
        self.lr = lr

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        K.set_learning_phase(1)
        K.set_session(self.sess)

        self.discriminator_loss_limit = discriminator_loss_limits

        self.actor = ActorNetwork(self.sess, self.cutoff, 3, self.batch_size, self.tau, self.lr)
        self.critic = CriticNetwork(self.sess, self.cutoff, 3, self.batch_size, self.tau, self.lr)

        self.discriminator = self._build_discriminator()
        self.discriminator_training_batch, self.discriminator_total_batch = 0, 0

        self.memory = Experience(memory_size=1000000, batch_size=self.batch_size, alpha=0.5)

        self.dataset_epoch = 0
        # self.actor.model.load_weights('actor')
        # self.actor.target_model.load_weights('actor')
        # self.critic.model.load_weights('critic')
        # self.critic.target_model.load_weights('critic')
        # self.discriminator.load_weights('discriminator')


    def _build_discriminator(self):

        state_input = Input(shape=(self.cutoff, 3))

        main_network = stacked_rnn(state_input, 125)

        discriminator_output = Dense(1, activation='sigmoid')(main_network)

        discriminator = Model(inputs=state_input, outputs=discriminator_output)
        discriminator.compile(optimizer=Adam(lr=self.lr), loss='binary_crossentropy')

        discriminator.summary()
        return discriminator

    # Main loop
    def train(self, epoch):

        e, total_frames = 0, 0
        while e <= epoch:

            discrim_loss = 0
            while discrim_loss >= self.discriminator_loss_limit:
                discrim_loss = self.train_discriminator()

            while self.memory.tree.size < self.min_history:
                self.add_data_to_memory_no_nstep()

            while True:

                if total_frames % 4 == 0:

                    self.add_data_to_memory_no_nstep()

                    if discrim_loss >= self.discriminator_loss_limit:
                        discrim_loss = self.train_discriminator()
                    else:
                        discrim_loss = self.train_discriminator(evaluate=True)
                self.train_on_replay_no_nstep()

                total_frames += 1

                if total_frames % 2000 == 0:
                    print('Epoch :', e,
                          '\tDataset Epoch :', self.dataset_epoch,
                          '\tDiscriminator training batch ratio :',
                          # '\t Average random weight', '%.4f' % self.actor.get_average_random_weight(),
                          '%.2f' % (self.discriminator_training_batch/self.discriminator_total_batch * 100.0), '%',
                          )
                    self.discriminator_total_batch, self.discriminator_training_batch = 0, 0
                    eval_seed, eval_loss = self.make_big_seed(5)
                    print('Eval loss :', '%.4f' % eval_loss, end='\t')
                    self.environnement.make_midi(eval_seed, str(e) + '.mid')
                    self.actor.model.save('saved_models/actor_' + str(e))
                    self.critic.model.save('saved_models/critic_' + str(e))
                    self.discriminator.save('saved_models/discriminator_' + str(e))
                    e += 1

    @nb.jit
    def train_discriminator(self, evaluate=False):
        fake_batch = self.get_fake_batch()
        real_batch, done = self.environnement.query_state(self.batch_size)
        if done is True:
            self.dataset_epoch += 1
        batch = np.vstack((real_batch, fake_batch))
        if evaluate is True:
            self.discriminator_total_batch += 1
            return self.discriminator.evaluate([batch], [self.labels], verbose=0)
        self.discriminator_total_batch += 1
        self.discriminator_training_batch += 1
        return self.discriminator.train_on_batch([batch], [self.labels])

    def get_fake_batch(self):
        seed = self.get_seed()
        states = np.zeros((self.batch_size + 1, self.cutoff, 3))
        states[0] = seed
        for i in range(self.batch_size):
            action = self.actor.target_model.predict(states[i:i+1])
            states[i+1, :-1] = states[i, 1:]
            states[i+1, -1] = action
        return states[:-1]

    def make_training_data_distributed(self):
        seed = self.get_seed()
        states = np.zeros((self.batch_size + self.n_steps + 1, self.cutoff, 3))
        actions = np.zeros((self.batch_size + self.n_steps + 1, 3))
        rewards = np.zeros((self.batch_size + self.n_steps + 1, 1))
        states[0] = seed
        for i in range(self.batch_size + self.n_steps):
            actions[i] = np.clip(self.actor.model.predict(states[i:i+1]) + np.random.normal(scale=1), -1, 1)
            states[i+1, :-1] = states[i, 1:]
            states[i+1, -1] = actions[i]
            rewards[i] = self.discriminator.evaluate(states[i+1:i+2], np.zeros((1,1)), verbose=0)
        critic_predictions = self.critic.target_model.predict([states[self.n_steps + 1:],
                                                               self.actor.target_model.predict(states[self.n_steps + 1:])])
        rewards = self.calc_rewards_distributed(rewards, critic_predictions)

        return states[:self.batch_size], actions[:self.batch_size], rewards[:self.batch_size]

    def make_training_data(self):
        seed = self.get_seed()
        states = np.zeros((self.batch_size + self.n_steps + 1, self.cutoff, 3))
        actions = np.zeros((self.batch_size + self.n_steps + 1, 3))
        rewards = np.zeros((self.batch_size + self.n_steps + 1, 1))
        states[0] = seed
        for i in range(self.batch_size + self.n_steps):
            actions[i] = np.clip(self.actor.model.predict(states[i:i+1]) + np.random.normal(), -1, 1)
            states[i+1, :-1] = states[i, 1:]
            states[i+1, -1] = actions[i]
            rewards[i] = self.discriminator.evaluate(states[i+1:i+2], np.zeros((1,1)), verbose=0)
        critic_predictions = self.critic.target_model.predict([states[self.n_steps + 1:],
                                                               self.actor.target_model.predict(states[self.n_steps + 1:])])
        rewards = self.calc_rewards(rewards, critic_predictions)

        return states[:self.batch_size], actions[:self.batch_size], rewards[:self.batch_size]

    def make_training_data_no_nstep(self):
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
        states, actions, rewards = self.make_training_data()
        for i in range(self.batch_size):
            self.memory.add((states[i], actions[i], rewards[i]), 10)

    def add_data_to_memory_no_nstep(self):
        states, actions, states_prime = self.make_training_data_no_nstep()
        for i in range(self.batch_size):
            self.memory.add((states[i], actions[i], states_prime[i]), 10)

    def add_data_to_memory_distributed(self):
        states, actions, rewards = self.make_training_data_distributed()
        for i in range(self.batch_size):
            self.memory.add((states[i], actions[i], rewards[i]), 10)

    @nb.jit
    def calc_rewards_distributed(self, rewards, predictions):
        m_prob = np.zeros(shape=predictions.shape)
        for i in range(self.batch_size):
            for j in range(self.n_steps):
                rewards[i] += (rewards[i + j + 1] * self.gammas[j])
        self.update_m_prob(rewards, m_prob, predictions)
        return m_prob[:self.batch_size]

    @nb.jit
    def calc_rewards(self, rewards, predictions):
        for i in range(self.batch_size):
            for j in range(self.n_steps):
                rewards[i] += (rewards[i + j + 1] * self.gammas[j])
        rewards[:self.batch_size] += predictions * self.gammas[-1]
        return rewards[:self.batch_size]

    @nb.jit
    def calc_rewards_no_nstep(self, states, states_primes):
        rewards = np.zeros((self.batch_size, 1))
        for i in range(self.batch_size):
            rewards[i] = self.discriminator.evaluate(states[i:i + 1], np.zeros((1, 1)), verbose=0)
            rewards[i,0] += self.critic.target_model.predict([states_primes[i:i+1],
                                    self.actor.target_model.predict(states_primes[i:i+1])]) * self.gammas[0]

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
        loss = np.mean([self.discriminator.evaluate(seed_list[i], np.zeros((1,1)), verbose=0) for i in range(times)])
        seed = np.concatenate(seed_list, axis=1)
        return seed, np.mean(loss)

    def make_dataset(self):
        data, weights, indices = self.memory.select(0.6)
        states, reward, actions = [], [], []
        for i in range(self.batch_size):
            states.append(data[i][0])
            actions.append(data[i][1])
            reward.append(data[i][2])
        states = np.array(states)
        reward = np.array(reward)
        actions = np.array(actions)
        return states, actions, reward, indices, weights

    def make_dataset_no_nstep(self):
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

    def train_on_replay_distributed(self):
        states, actions, rewards, indices, weights = self.make_dataset()
        loss = self.critic.model.train_on_batch([states, actions], rewards)

        a_for_grad = self.actor.model.predict(states)
        grads = self.critic.gradients(states, a_for_grad)
        self.actor.train(states, grads)
        self.actor.target_train()
        self.critic.target_train()
        self.memory.priority_update(indices, [loss for _ in range(len(indices))])

    def train_on_replay_no_nstep(self):
        states, actions, states_prime, indices = self.make_dataset_no_nstep()
        rewards = self.calc_rewards_no_nstep(states, states_prime)
        loss = self.critic.model.train_on_batch([states, actions], rewards)

        a_for_grad = self.actor.model.predict(states)
        grads = self.critic.gradients(states, a_for_grad)

        self.actor.train(states, grads)
        self.actor.target_train()
        self.critic.target_train()
        self.memory.priority_update(indices, [loss for _ in range(len(indices))])

    @nb.jit
    def update_m_prob(self, reward, m_prob, z):
        for i in range(self.batch_size):
            for j in range(self.atoms):
                tz = min(self.v_max, max(self.v_min, reward[i] + self.gammas[-1] * z[i, j]))
                bj = (tz - self.v_min) / self.delta_z
                m_l, m_u = math.floor(bj), math.ceil(bj)

                if m_l == m_u == self.atoms - 1:
                    m_l -= 1
                elif m_l == m_u == 0:
                    m_u += 1

                m_prob[i,  int(m_l)] += z[i, j] * (m_u - bj)
                m_prob[i,  int(m_u)] += z[i, j] * (bj - m_l)


if __name__ == '__main__':
    agent = Agent(cutoff=30, discriminator_loss_limits=0.25, batch_size=64)
    agent.train(epoch=5000)
