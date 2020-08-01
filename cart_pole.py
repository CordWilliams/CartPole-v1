import gym
import numpy as np
import random

from collections import deque
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import adam
from keras.activations import relu, linear

env = gym.make('CartPole-v1')
env.seed(1)
np.random.seed(1)


class CartPoleSolver:
    def __init__(self, actions, states, e, g, batch_len, e_min, a, e_decay, mem_size):
        self.num_of_actions = actions
        self.num_of_states = states
        self.epsilon = e
        self.gamma = g
        self.batch_size = batch_len
        self.epsilon_min = e_min
        self.alpha = a
        self.epsilon_decay = e_decay
        self.memory = deque(maxlen=mem_size)
        self.model = self.initialize_q_function()

    def initialize_q_function(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.num_of_states, activation=relu))
        model.add(Dense(256, activation=relu))
        model.add(Dense(self.num_of_actions, activation=linear))
        model.compile(loss='mse', optimizer=adam(lr=self.alpha))
        return model

    def get_action(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.num_of_actions)
        return np.argmax(self.model.predict(state)[0])

    def save_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_experience(self):
        if len(self.memory) < self.batch_size:
            return

        mini_batch_of_samples = random.sample(self.memory, self.batch_size)
        state_samples = np.squeeze(np.array([i[0] for i in mini_batch_of_samples]))
        action_samples = np.array([i[1] for i in mini_batch_of_samples])
        reward_samples = np.array([i[2] for i in mini_batch_of_samples])
        next_state_samples = np.squeeze(np.array([i[3] for i in mini_batch_of_samples]))
        done_samples = np.array([i[4] for i in mini_batch_of_samples])

        targets = reward_samples + self.gamma * (np.amax(self.model.predict_on_batch(next_state_samples), axis=1)) * (
                1 - done_samples)
        targets_full = self.model.predict_on_batch(state_samples)
        targets_full[[(np.array([i for i in range(self.batch_size)]))], [action_samples]] = targets
        self.model.fit(state_samples, targets_full, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_learner(self, episodes):
        loss = []
        for episode in range(episodes):
            s = np.reshape(env.reset(), (1, 4))
            total_reward = 0
            i = 0
            done = False
            while not done:
                env.render()
                a = self.get_action(s)
                s_prime, reward, done, info = env.step(a)
                total_reward += reward
                s_prime = np.reshape(s_prime, (1, 4))
                self.save_experience(s, a, reward, s_prime, done)
                s = s_prime
                self.replay_experience()
                i += 1
            loss.append(total_reward)

            print(np.mean(loss[-100:]))
            if np.mean(loss[-100:]) > 200:
                print("You win after " + str(episode) + " episodes.")
                break

        env.close()
        return loss


if __name__ == '__main__':
    num_of_actions = env.action_space.n
    num_of_states = env.observation_space.shape[0]
    epsilon = 1.0
    gamma = .99
    batch_size = 64
    epsilon_min = .01
    alpha = 0.0001
    epsilon_decay = .996
    mem_len = 1000000
    num_of_episodes = 50000

    learner = CartPoleSolver(num_of_actions, num_of_states, epsilon, gamma, batch_size, epsilon_min, alpha, epsilon_decay,
                             mem_len)
    learner.train_learner(num_of_episodes)
    print("Solve CartPole-v1 using DQN")
