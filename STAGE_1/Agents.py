__author__ = 'Dylan Klein'
'''This file defines the class for creating an agent who learns via the Q-Learning algorithm'''

from STAGE_1.Naming_Convention import integer_to_letter as int2let
from random import *


class Q_Learner():

    def __init__(self, agent_id, discount, exploit_period, q, env):

        self.agent_id = agent_id  # Numerical ID for each agent
        self.discount = discount  # Discount factor
        self.exploit_period = exploit_period  # After this many actions are taken, the agent will take a random action
        self.exploit_counter = 0  # To track when an agent will execute a random action
        self.Q = q  # The current Q-table of the agent
        self.env = env
        self.action = "none"
        self.alpha = 1  # The agent's learning rate
        self.cumulative_reward = 0  # The agent's total reward

        # The initial Q-table of the agent is to be loaded from file if input is not {}
        if self.Q != {}:
            string = 'agent' + int2let(agent_id) + '_Q.txt'
            with open(string, 'r') as f:
                self.Q = f.read()

        # If input is {}, then create a new Q-table, with all Q-values initialised to 0.1
        else:
            for state in env.state_space:
                temp = {}
                for action in env.actions:
                    temp[action] = 0.1
                self.Q[state] = temp  # Initialise Q table

    # Decide on the best action to take (with the exception of a random action now and again)
    def step(self, time, s):

        # Evaluate the best action
        max_act, max_val = Q_Learner.max_Q(self, s)

        # Do the best action
        if self.exploit_counter < self.exploit_period:
            self.exploit_counter += 1  # Increment the counter
            self.action = max_act

        # Do a random action
        else:
            self.exploit_counter = 0  # Reset the counter
            self.action = self.env.actions[randint(0, 4)]

        '''THE AGENT MUST INCLUDE THE FOLLOWING LINE: A STEP IN THE ENVIRONMENT'''
        # Take the selected action
        (s2, r, dn, ep) = self.env.step(self.agent_id-1, self.action)

        # Update Q
        max_act, max_val = Q_Learner.max_Q(self, s2)
        Q_Learner.inc_Q(self, s, self.action, self.alpha, r + self.discount * max_val)

        # Update the learning rate
        self.alpha = pow(time, -0.1)

        # Update cumulative reward
        self.cumulative_reward += r

        return s2, dn, ep

# Find the maximum Q-value and action pair for a given state
    def max_Q(self, s):
        val = None
        act = None
        for a, q in self.Q[s].items():
            if val is None or (q > val):
                val = q
                act = a
        return act, val

    # Update the Q-value based on the learning rate
    def inc_Q(self, s, a, alpha, inc):
        self.Q[s][a] *= 1 - alpha
        self.Q[s][a] += alpha * inc

