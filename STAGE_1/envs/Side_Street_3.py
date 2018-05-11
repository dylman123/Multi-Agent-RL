__author__ = 'Dylan Klein'
'''This file is the main file to execute when training multiple agents to negotiate via RL'''

from STAGE_1.Make_World import Grid_World
from STAGE_1.Agents import Q_Learner
import threading
import time
from random import *


''' 
    *** DEFINE THE WALL CELLS OF THE WORLD ***
        'walls' is defined as a list with entries in the following syntax:
            (x, y) where
                x = x_coordinate of cell
                y = y_coordinate of cell
'''
walls = [(0, 0), (1, 0), (3, 0), (4, 0), (5, 0), (6, 0),
         (0, 2), (1, 2), (2, 2), (3, 2), (5, 2), (6, 2)]


'''
    *** DEFINE THE SPECIAL CELLS OF THE WORLD ***
        'specials' is defined as a list with entries in the following syntax:
            (a_id, (x, y), c, r) where
                a_id = agent_id (if a_id = 0, this indicates that the special cell is common to all agents)
                x = x_coordinate of cell
                y = y_coordinate of cell
                c = colour of cell (red or green)
                r = reward of cell
'''
specials = [(1, (6, 1), "green", 5), (2, (0, 1), "green", 5), (1, (4, 2), "red", -5), (2, (2, 0), "red", -5)]


'''
    *** CREATE AN ENVIRONMENT OBJECT ***
        - width and height are the dimensions of the grid world
        - initial_state is defined as a tuple with agents' initial position coordinates written in order,
        for example if Agent A starts at (0, 0) and Agent B starts at (2, 4) then initial_state = (0, 0, 2, 4)
'''
init_state = (0, 1, 6, 1)
env = Grid_World(width=7, height=3, initial_state=init_state, walls=walls, specials=specials)


''' 
    *** INITIALISE THE ENVIRONMENT ***
'''
observation = env.reset()


''' 
    *** CREATE MULTIPLE AGENT OBJECTS *** 
'''

# Create instances of agents from the Q_Learner class
AgentA = Q_Learner(agent_id=1, discount=0.3, exploit_period=50, q={}, env=env)
AgentB = Q_Learner(agent_id=2, discount=0.3, exploit_period=50, q={}, env=env)


'''
    *** MAIN PROGRAM ***
    - Agents take steps in a random order
    - rewards per step are held in the array 'rewards'
    - cumulative rewards per episode are held in the array 'avg_cumulative_rewards'
'''


# Agents each decide on an individual action to take in the next step
def step_all_agents(t):
    global observation, done, episode

    i = randint(0, env.num_agents - 1)  # Randomise the order of agents' turns
    for n in range(env.num_agents):
        if i is 0:
            observation, done, episode = AgentA.step(time=t, s=observation)
            i = 1
        elif i is 1:
            observation, done, episode = AgentB.step(time=t, s=observation)
            i = 0


def run():
    global observation, done, episode

    # Variable time steps to speed up game play
    med_step = 0.1
    slow_step = 1
    view_mode = 500  # After this many episodes, slow game play to viewing speeds

    # Initialise loop variables
    done = False
    episode = 0
    t = 1

    while episode < 5000:

        # If episode has ended, reset env and print results
        if done is True:
            env.render()
            observation = env.reset()
            t = 1

            # Output learning results to command window
            print(episode, AgentA.cumulative_reward / episode, AgentB.cumulative_reward / episode)

            # Pause to show end of episode
            if episode > view_mode:
                time.sleep(slow_step)

        env.render()

        # Regular runtime in each episode
        step_all_agents(t)
        t += 1

        # For easy viewing speeds
        if episode > view_mode:
            time.sleep(med_step)


tk = threading.Thread(target=run)
tk.daemon = True
tk.start()
env.start_game()
