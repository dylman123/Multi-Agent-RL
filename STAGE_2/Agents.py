__author__ = 'Dylan Klein'
'''This file defines the class for creating agents who learn to navigate and negotiate in a multi-agent environment'''

from Naming_Convention import integer_to_letter as int2let
from random import *
import copy
import pickle
import os
import datetime
import tensorflow as tf


# For time stamping directories when agents Q-functions are saved
def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


class Q_Table:

    def __init__(self, agent_id, discount, epsilon_decay, states, actions, q):

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
        self.states = states  # The entire state space of the game
        self.state = []  # An agent's previous state
        self.state2 = []  # An agent's new state
        self.arrow = {}  # Object to store the agent's intent arrows

        # The initial Q-table can be either loaded from file or created from scratch
        if self.Q == "load":
            Q_Table.load(self)
        elif self.Q == "new":
            Q_Table.new(self)

    # Load Q-table from file
    def load(self):
        with open('Saved_Files/' + 'agent' + int2let(self.agent_id+1) + '_saved' + '.pkl', 'rb') as f:
            self.Q = pickle.load(f)

    # Save Q-table to file
    def save(self):
        stamp = timeStamped("")
        os.makedirs('Saved_Files/' + stamp)
        with open('Saved_Files/' + stamp + '/agent' + int2let(self.agent_id+1) + '_saved' + '.pkl', 'wb') as f:
            pickle.dump(self.Q, f, pickle.HIGHEST_PROTOCOL)

    # Create a new Q-table, with all Q-values initialised to 0.1
    def new(self):
        self.Q = {}
        for state in self.states:
            temp = {}
            for action in self.actions:
                temp[action] = 0.1
            self.Q[state] = temp  # Initialise Q table

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
    def learn(self, t, restart):

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
        self.alpha = pow(t, -0.1)

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

    def __init__(self, agent_id, discount, epsilon_decay, input_size, actions, q):

        self.agent_id = agent_id  # Numerical ID for each agent
        self.discount = discount  # Discount factor
        self.epsilon = 1  # Initial value of agent's epsilon
        self.epsilon_decay = epsilon_decay  # How much epsilon decays per step
        self.input_size = input_size  # Input the size of the environment's state space
        self.actions = actions  # Input the environment's action space
        self.Q = q
        self.action = "none"

        tf.reset_default_graph()
        self.num_actions = len(self.actions)

        # The initial DQN can be either loaded from file or created from scratch
        if self.Q == "load":
            DQN.load(self)
        elif self.Q == "new":
            DQN.new(self)

    # Load DQN from file
    def load(self):
        with open('Saved_Files/' + 'agent' + int2let(self.agent_id + 1) + '_saved' + '.pkl', 'rb') as f:
            self.Q = pickle.load(f)

    # Save Q-table to file
    def save(self):
        stamp = timeStamped("")
        os.makedirs('Saved_Files/' + stamp)
        with open('Saved_Files/' + stamp + '/agent' + int2let(self.agent_id + 1) + '_saved' + '.pkl',
                  'wb') as f:
            pickle.dump(self.Q, f, pickle.HIGHEST_PROTOCOL)

    # Create a new DQN
    def new(self):
        # These lines establish the feed-forward part of the network used to choose actions
        inputs1 = tf.placeholder(shape=[1, self.input_size], dtype=tf.float32)
        W = tf.Variable(tf.random_uniform([self.input_size, self.num_actions], 0, 0.01))
        Qout = tf.matmul(inputs1, W)
        predict = tf.argmax(Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q-values.
        nextQ = tf.placeholder(shape=[1, self.num_actions], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(nextQ - Qout))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        updateModel = trainer.minimize(loss)

    # Decide on the best action to take (with the exception of a random action now and again)
    def act(self):

        # Evaluate the best action

        # Do a random action
        if random() < self.epsilon:
            self.action = self.actions[randint(0, 4)]

        # Do the best action
        else:
            self.action = max_act

    # Learn from the new state and reward pair as updated by the environment
    def learn(self, t, restart):

        # Decay epsilon value at the end of each episode
        if restart is True and self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay

