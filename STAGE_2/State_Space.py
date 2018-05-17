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

Both of these state encoding examples are for ONE INDIVIDUAL AGENT only.

In a 2-agent game:
If relative coords is selected, then an example state might be (0, 0, -2, -2, 1, 0, 0, 1) which means: Agent A has
no concept of where it is located but it knows that it itself is located at its own origin (0, 0). It also knows
that Agent B is located 2 cells up and 2 cells left from itself due to the relative coordinates of Agent B (-2, -2)
from Agent A's origin. Like in the other examples, its possible goals are the first and fourth due to the
one-hot encoding 1001.

If Q_Table is not selected as the learning algorithm, the state space can be much larger since it does not need
to be explicitly defined. Therefore if neural networks are used instead of a Q_Table, then the relative state space
format will be:
(0, 0, relative coordinates of other agents, relative coordinates of all the possible goals, one-hot coding of 
desired goals), for example:

(0, 0, -2, 3, 4, 0, 1, -1, 2, 2, -1, 1, 0, 0, 1, 0) is a 2-agent world with 4 goals and one-hot coding 0010.
'''


# This function generates a grid state space with multiple agents (1-5)
def make_states(num_agents, coords_type, num_goals, x, y):

    state_space = []
    goal_table = list(itertools.product([0, 1], repeat=num_goals))  # Goal possibilities in one-hot encoding

    if coords_type == "relative":
        x_range = range(-x + 1, x)
        y_range = range(-y + 1, y)

    else:  # if coords_type == "absolute":
        x_range = range(x)
        y_range = range(y)

    # Create a large state space using FOR loop iterations
    if num_agents == 1:
        for a in x_range:
            for b in y_range:
                    for goal in goal_table:
                        state_space.append(tuple([a, b] + list(goal)))

    elif num_agents == 2:
        for a in x_range:
            for b in y_range:
                for c in x_range:
                    for d in y_range:
                            for goal in goal_table:
                                state_space.append(tuple([a, b, c, d] + list(goal)))

    elif num_agents == 3:
        for a in x_range:
            for b in y_range:
                for c in x_range:
                    for d in y_range:
                        for e in x_range:
                            for f in y_range:
                                    for goal in goal_table:
                                        state_space.append(tuple([a, b, c, d, e, f] + list(goal)))

    elif num_agents == 4:
        for a in x_range:
            for b in y_range:
                for c in x_range:
                    for d in y_range:
                        for e in x_range:
                            for f in y_range:
                                for g in x_range:
                                    for h in y_range:
                                        for goal in goal_table:
                                            state_space.append(tuple([a, b, c, d, e, f, g, h] + list(goal)))
    return state_space
