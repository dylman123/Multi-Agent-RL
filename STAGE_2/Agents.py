__author__ = 'Dylan Klein'
'''This file defines the class for creating agents who learn to navigate and negotiate in a multi-agent environment'''

from Naming_Convention import integer_to_letter as int2let
from random import *
import copy
import pickle
# import tensorflow as tf


class Q_Table:

    def __init__(self, agent_id, discount, epsilon_decay, actions, q):

        self.agent_id = agent_id  # Numerical ID for each agent
        self.discount = discount  # Discount factor
        self.alpha = 0.1  # The agent's learning rate
        self.epsilon = 1  # Initial value of agent's epsilon
        self.epsilon_decay = epsilon_decay  # How much epsilon decays per step
        self.Q = copy.deepcopy(q)  # The Q-table of the agent
        self.actions = actions  # Input the environment's action space
        self.action = "none"  # The agent's chosen action for a single step
        self.goal = (0, 0, 0, 0)  # An agent's goal in one-hot coding
        self.intent = "none"  # An agent's intended goal in words
        self.position = (0, 0)  # An agent's coordinates
        self.reward = 0  # An agent's reward per step
        self.state = []  # An agent's previous state
        self.state2 = []  # An agent's new state
        self.arrow = {}

        # The initial Q-table of the agent is to be loaded from file if input is not {}
        if self.Q == "load":
            Q_Table.load(self)

    # Load Q-table from file
    def load(self):
        with open('Saved_Files/' + 'agent' + int2let(self.agent_id+1) + '_saved' + '.pkl', 'rb') as f:
            self.Q = pickle.load(f)

    # Save Q-table to file
    def save(self):
        with open('Saved_Files/' + 'agent' + int2let(self.agent_id+1) + '_saved' + '.pkl', 'wb') as f:
            pickle.dump(self.Q, f, pickle.HIGHEST_PROTOCOL)

    # Decide on the best action to take (with the exception of a random action now and again)
    def act(self):

        # Evaluate the best action
        max_act, max_val = Q_Table.max_Q(self, self.state)

        # Do a random action
        if random() < self.epsilon:
            self.action = self.actions[randint(0, 4)]

        # Do the best action
        else:
            self.action = max_act

    # Learn from the new state and reward pair as updated by the environment
    def learn(self, time, restart):

        # Use shorthand notation for readability
        s = self.state
        s2 = self.state2
        a = self.action
        r = self.reward
        y = self.discount
        lr = self.alpha
        Q = self.Q

        # Find the maximum value for the new state
        max_act, max_val = Q_Table.max_Q(self, s2)

        # Update Q-Table using the Bellman Equation
        self.Q[s][a] = Q[s][a] + lr * (r + y * max_val - Q[s][a])

        # Update the learning rate
        self.alpha = pow(time, -0.1)

        # Decay epsilon value at the end of each episode
        if restart is True and self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay

    # Find the maximum Q-value and action pair for a given state
    def max_Q(self, s):
        val = None
        act = None
        for a, q in self.Q[s].items():
            if val is None or (q > val):
                val = q
                act = a
        return act, val


class DQN:

    def __init__(self, agent_id, discount, epsilon_decay, actions, num_states, q):

        self.agent_id = agent_id  # Numerical ID for each agent
        self.discount = discount  # Discount factor
        self.epsilon_decay = epsilon_decay  # How much epsilon decays per step
        self.actions = actions  # Input the environment's action space
        self.num_states = num_states  # Input the size of the environment's state space

        # These lines establish the feed-forward part of the network used to choose actions
        num_actions = len(self.actions)
        inputs1 = tf.placeholder(shape=[1, self.num_states], dtype=tf.float32)
        W = tf.Variable(tf.random_uniform([self.num_states, num_actions], 0, 0.01))
        Qout = tf.matmul(inputs1, W)
        predict = tf.argmax(Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q-values.
        nextQ = tf.placeholder(shape=[1, num_actions], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(nextQ - Qout))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        updateModel = trainer.minimize(loss)
