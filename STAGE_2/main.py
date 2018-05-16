__author__ = 'Dylan Klein'
'''This file is the main file to execute when training multiple agents to negotiate via RL'''


from Make_World import World
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

env = World(map_type="plus", num_agents=3, agent_type="Q_Table", coords_type="absolute", load="no", save="no")
print(len(env.states))

# Save agents' Q-tables or Neural Networks to file
def save_agents():
    if env.save is "yes" and env.load is "no":
        env.write_to_file()


# *** TESTING ***
def start_testing():
    env.epsilon_greedy()  # Sets agents' epsilon values to 0 for testing
    while True:
        env.render()
        env.step()
        time.sleep(0.8)


# *** TRAINING ***
def start_training():
    try:
        # Initialise loop variables
        cumulative_rewards = [0] * env.num_agents

        while True:
            ep = env.episode_count

            # Take a step in the environment
            observation, rewards, done, info = env.step()

            # Calculate the agents' cumulative rewards
            cumulative_rewards = np.array(cumulative_rewards) + np.array(rewards)

            # Print the agents' average reward per episode
            if done is True and ep % 100 is 0:
                pass  #print(ep, cumulative_rewards / ep)

    # A keyboard interrupt will exit training mode
    except KeyboardInterrupt:
        save_agents()

    finally:
        tk = threading.Thread(target=start_testing)
        tk.daemon = True
        tk.start()
        env.start_game()


'''
    *** MAIN PROGRAM ***
    - Agents take steps in a random order
    - 'observation' is the variable to hold global state information
    - 'done' is the flag to indicate if the episode has ended
    - 'rewards' is a list of each agents reward per step
    - 'info' stores any extra episode information
'''

start_training()
