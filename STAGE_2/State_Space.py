__author__ = 'Dylan'
'''This file creates the state space for use in agents' Q-tables'''

import itertools

'''
The state of an individual agent will be formatted as such:
(agent's own coordinates, other agent's coordinates, agent's own goal in one-hot encoding)

For example, (3, 2, 1, 0, 1, 0) means that this is a single agent world where the agent's coordinates are
(3, 2) and the agent's individual goal/intent in one-hot encoding is 1001. This code means that the agent would
be rewarded were it to land on EITHER the first or third goal cell, and not be rewarded on the second or fourth.

Another example may be (0, 4, 1, 1, 0, 0, 0, 1) which is interpreted as: Agent A is located at (0, 4) and
Agent B is located at (1, 1). Agent A's individual goal in one-hot encoding is 0001. This means that Agent A
can only be rewarded by landing on the fourth goal cell.

Both of these state eoncoding examples are for ONE INDIVIDUAL AGENT only.
'''


# This function generates a grid state space with multiple agents (1-5)
def make_states(num_agents, num_goals, x, y):

    state_space = []
    goal_table = list(itertools.product([0, 1], repeat=num_goals))  # Goal possibilities in one-hot encoding

    if num_agents == 1:
        for a in range(x):
            for b in range(y):
                for goal in goal_table:
                    state_space.append(tuple([a, b] + list(goal)))

    elif num_agents == 2:
        for a in range(x):
            for b in range(y):
                for c in range(x):
                    for d in range(y):
                        for goal in goal_table:
                            state_space.append(tuple([a, b, c, d] + list(goal)))

    elif num_agents == 3:
        for a in range(x):
            for b in range(y):
                for c in range(x):
                    for d in range(y):
                        for e in range(x):
                            for f in range(y):
                                for goal in goal_table:
                                    state_space.append(tuple([a, b, c, d, e, f] + list(goal)))

    elif num_agents == 4:
        for a in range(x):
            for b in range(y):
                for c in range(x):
                    for d in range(y):
                        for e in range(x):
                            for f in range(y):
                                for g in range(x):
                                    for h in range(y):
                                        for goal in goal_table:
                                            state_space.append(tuple([a, b, c, d, e, f, g, h] + list(goal)))

    return state_space
