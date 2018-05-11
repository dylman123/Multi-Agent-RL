__author__ = 'Dylan'
'''This file creates the state space for use in agents' Q-tables'''


# This function generates a grid state space with multiple agents (1-5)
def make_states(n, x, y):

    state_space = []

    if n == 1:
        for a in range(-x+1, x):
            for b in range(-y+1, y):
                state_space.append((a, b))

    elif n == 2:
        for a in range(-x+1, x):
            for b in range(-y+1, y):
                for c in range(-x+1, x):
                    for d in range(-y+1, y):
                        state_space.append((a, b, c, d))

    elif n == 3:
        for a in range(-x + 1, x):
            for b in range(-y + 1, y):
                for c in range(-x + 1, x):
                    for d in range(-y + 1, y):
                        for e in range(-x + 1, x):
                            for f in range(-y + 1, y):
                                state_space.append((a, b, c, d, e, f))

    elif n == 4:
        for a in range(-x + 1, x):
            for b in range(-y + 1, y):
                for c in range(-x + 1, x):
                    for d in range(-y + 1, y):
                        for e in range(-x + 1, x):
                            for f in range(-y + 1, y):
                                for g in range(-x + 1, x):
                                    for h in range(-y + 1, y):
                                        state_space.append((a, b, c, d, e, f, g, h))

    elif n == 5:
        for a in range(-x + 1, x):
            for b in range(-y + 1, y):
                for c in range(-x + 1, x):
                    for d in range(-y + 1, y):
                        for e in range(-x + 1, x):
                            for f in range(-y + 1, y):
                                for g in range(-x + 1, x):
                                    for h in range(-y + 1, y):
                                        for i in range(-x + 1, x):
                                            for j in range(-y + 1, y):
                                                state_space.append((a, b, c, d, e, f, g, h, i, j))

    return state_space
