__author__ = 'Dylan Klein'
'''This file is the main file to execute when training multiple agents to negotiate via RL'''


from STAGE_2.Make_World import World
import numpy as np
import threading
import time


'''
    *** CREATE AN ENVIRONMENT OBJECT ***
    - 'map_type' defines the geometry of the world
    - 'num_agents' defines the number of agents in the environment
    - 'agent_type' defines the algorithm driving each agent
    - load="no": creates a new neural network for agents, load="yes": loads neural networks from file
    - save="no": does not save agents' neural networks to file, save="yes": saves neural networks to file
'''

env = World(map_type="plus", num_agents=2, agent_type="Q_Table", load="no", save="no")

'''
    *** MAIN PROGRAM ***
    - Agents take steps in a random order
    - 'observation' is the variable to hold global state information
    - 'done' is the flag to indicate if the episode has ended
    - 'rewards' is a list of each agents reward per step
    - 'info' stores any extra episode information
'''


def run():

    # Initialise loop variables
    cumulative_rewards = np.array([0] * env.num_agents)
    window_raw = [0] * 50
    window_smooth = [0] * 50
    converged = False

    # *** Training ***
    while converged is False:
        ep = env.episode_count

        # Take a step in the environment
        observation, rewards, done, info = env.step()

        # Calculate the agents' cumulative rewards
        cumulative_rewards = cumulative_rewards + np.array(rewards)

        # Check for convergence
        window_raw = shift(window_raw, -1)
        window_raw[0] = cumulative_rewards[0] / ep
        smoothed_data = sum(window_raw) / len(window_raw)

        # window_smooth =
        # if has_converged(window_smooth):
        #  pass  # converged = True

        # Print the agents' average reward per episode
        if done is True and ep % 1 is 0:
            print(cumulative_rewards[0] / ep, smoothed_data)

    # Save agents' Q-tables or Neural Networks to file
    if env.save is "yes" and env.load is "no":
        env.write_to_file()

    # *** Testing ***
    env.test()
    while True:
        env.render()
        env.step()
        time.sleep(0.8)


# Shift list
def shift(l, n):
    return l[n:] + l[:n]


# Mathematical check for convergence
def has_converged(window):
    upper_lower_difference = list((np.array(window[-1]) - np.array(window[0])))
    for i in range(len(upper_lower_difference)):
        if abs(upper_lower_difference[i]) < 0.00000001:
            # print("Converged!")
            return True
    return False


tk = threading.Thread(target=run)
tk.daemon = True
tk.start()
env.start_game()
