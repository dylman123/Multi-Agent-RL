__author__ = 'Dylan Klein'
'''This file defines the class for creating grid world environments as an object'''

from State_Space import make_states
from Agents import *
from tkinter import *  # For rendering
from random import randint, choice
import numpy as np

# import matplotlib.pyplot as plt


class World:

    def __init__(self, map_type, coords_type, num_agents, agent_type, load, save):

        # File saving
        self.load = load  # If load has value "yes", read each agent's Q-table from a file for initialisation
        self.save = save  # If save has value "yes", write each agent's Q-table to a file after training is completed

        # Environment variables
        self.actions = ["up", "down", "left", "right", "none"]
        self.wall_punishment = 1
        self.walk_punishment = 0.04
        self.goal_reward = 5  # To give if all agents collaborate
        self.goal_count = 0  # To track how many goals have been reached (in episodic mode only)
        self.crash_punishment = 10  # To punish if any agents crash
        self.width, self.height, self.walls, self.goals, self.starts, self.map_mode = World.create_map(self, map_type)
        self.coords_type = coords_type  # To toggle between relative and absolute coordinates

        # Create a board to render to
        self.master = Tk()
        self.L = 100  # side length of a cell in pixels
        self.board = Canvas(self.master, width=self.width * self.L, height=self.height * self.L)
        self.board.grid(row=0, column=0)

        # Render the static components of the environment
        for i in range(self.width):
            for j in range(self.height):
                self.board.create_rectangle(i * self.L, j * self.L, (i + 1) * self.L, (j + 1) * self.L, fill="white", width=1)
        for ((i, j), g_id) in self.goals:
            self.board.create_rectangle(i * self.L, j * self.L, (i + 1) * self.L, (j + 1) * self.L, fill="green", width=1)
        for (i, j) in self.walls:
            self.board.create_rectangle(i * self.L, j * self.L, (i + 1) * self.L, (j + 1) * self.L, fill="black", width=1)

        # Multi-agent variables
        self.num_agents = num_agents
        self.states = make_states(self.num_agents, self.width, self.height)  # Create the entire state space
        self.num_states = len(self.states)
        self.agent_list = World.make_agents(self, agent_type)  # To create instances of agents
        self.collisions = 0  # To count how many collisions have occurred in each episode

        # RL variables
        self.global_state = [0] * (self.num_agents * 2)  # Create a global state array
        self.rewards = 0
        self.restart = False
        self.episode_info = ""
        self.episode_count = 1  # Integer to store how many episodes have occurred
        self.time_step = 0  # To count each step

        # Create an initial random state
        World.reset_all_agents(self)

        # Create objects for rendering agents
        self.colours = ["orange", "blue", "pink", "purple", "yellow"]
        self.names = ["A", "B", "C", "D", "E"]
        self.triangle_size = 0.2
        for agent in self.agent_list:
            (x, y) = agent.position
            agent.object = self.board.create_rectangle(x * self.L + self.L * 2 / 10,
                                                       y * self.L + self.L * 2 / 10,
                                                       x * self.L + self.L * 8 / 10,
                                                       y * self.L + self.L * 8 / 10,
                                                       fill=self.colours[agent.agent_id],
                                                       width=1, tag=self.names[agent.agent_id])

            agent.arrow = self.board.create_polygon(0, 0, fill="black", width=1)

    # Start rendering the 'master' object
    def start_game(self):
        self.master.mainloop()

    def reset_all_agents(self):
        for agent in self.agent_list:
            World.reset(self, agent)

    def reset(self, agent):
        World.spawn(self, agent)
        World.new_goal(self, agent)
        World.update_intent(self, agent)
        World.update_global_state(self)  # Update the global state

    def render(self):
        for agent in self.agent_list:
            (x, y) = agent.position
            self.board.coords(agent.object,
                              x * self.L + self.L * 2 / 10,
                              y * self.L + self.L * 2 / 10,
                              x * self.L + self.L * 8 / 10,
                              y * self.L + self.L * 8 / 10)
            World.update_arrow(self, agent)

    def update_arrow(self, agent):
        (x, y) = agent.position
        nudge = self.triangle_size / 2
        if agent.intent == "S":
            (self.board.coords(agent.arrow,
                               (x + 0.5 - self.triangle_size) * self.L, (y + 0.5 - nudge) * self.L,
                               (x + 0.5 + self.triangle_size) * self.L, (y + 0.5 - nudge) * self.L,
                               (x + 0.5) * self.L, (y + 0.5 + self.triangle_size - nudge) * self.L))
        elif agent.intent == "W":
            (self.board.coords(agent.arrow,
                               (x + 0.5 + nudge) * self.L, (y + 0.5 - self.triangle_size) * self.L,
                               (x + 0.5 + nudge) * self.L, (y + 0.5 + self.triangle_size) * self.L,
                               (x + 0.5 + nudge - self.triangle_size) * self.L, (y + 0.5) * self.L))
        elif agent.intent == "N":
            (self.board.coords(agent.arrow,
                               (x + 0.5 - self.triangle_size) * self.L, (y + 0.5 + nudge) * self.L,
                               (x + 0.5 + self.triangle_size) * self.L, (y + 0.5 + nudge) * self.L,
                               (x + 0.5) * self.L, (y + 0.5 - self.triangle_size + nudge) * self.L))
        elif agent.intent == "E":
            (self.board.coords(agent.arrow,
                               (x + 0.5 - nudge) * self.L, (y + 0.5 - self.triangle_size) * self.L,
                               (x + 0.5 - nudge) * self.L, (y + 0.5 + self.triangle_size) * self.L,
                               (x + 0.5 - nudge + self.triangle_size) * self.L, (y + 0.5) * self.L))

    def step(self):
        self.time_step += 1
        self.rewards = [0] * self.num_agents  # Reset the rewards list to zero
        self.restart = False
        copy_agent_list = list(self.agent_list)

        for _ in range(self.num_agents):
            agent = choice(copy_agent_list)  # Choose a random agent from the list
            copy_agent_list.remove(agent)  # Remove agent from the list so it can't be selected again in the same step

            # Initialise the agent's reward for the step
            agent.reward = 0

            # Reformat state from global to local (per agent)
            agent.state = World.reformat_state(self, agent, self.coords_type)

            # Agent to choose an action for the step
            agent.act()

            # Calculate incremental changes in x,y coordinates
            if agent.action == self.actions[0]:
                (dx, dy) = (0, -1)  # UP
            elif agent.action == self.actions[1]:
                (dx, dy) = (0, 1)  # DOWN
            elif agent.action == self.actions[2]:
                (dx, dy) = (-1, 0)  # LEFT
            elif agent.action == self.actions[3]:
                (dx, dy) = (1, 0)  # RIGHT
            else:
                (dx, dy) = (0, 0)  # NONE

            # Calculate the new proposed position
            new_x = agent.position[0] + dx
            new_y = agent.position[1] + dy

            # Try move the caller agent to a new cell if possible
            if (new_x >= 0) and (new_x < self.width) and (new_y >= 0) and (new_y < self.height) and not ((new_x, new_y) in self.walls):
                agent.position = (new_x, new_y)
                agent.reward = -self.walk_punishment

                # Update the global state
                World.update_global_state(self)

                # Check for a collision of agents
                if World.has_collided(self, agent) is True:
                    agent.reward = -self.crash_punishment
                    if self.map_mode == "episodic":
                        World.reset_all_agents(self)
                    elif self.map_mode == "non-episodic":
                        World.reset(self, agent)
                    self.collisions += 1
                    self.episode_count += 1
                    self.restart = True
                else:
                    # Check for landing on a goal
                    for ((x, y), g_id) in self.goals:
                        if agent.position == (x, y) and agent.goal == g_id:
                            agent.reward = self.goal_reward
                            if self.map_mode == "episodic":
                                World.reset(self, agent)
                            elif self.map_mode == "non-episodic":
                                World.new_goal(self, agent)  # Find a new goal for the agent
                                World.update_intent(self, agent)
                            self.episode_count += 1
                            self.restart = True

            # If the cell is not valid, it must belong to a wall our reside outside the world boundaries
            else:
                agent.reward = -self.crash_punishment

            # Reformat state from global to local (per agent)
            agent.state2 = World.reformat_state(self, agent, self.coords_type)

            # Agent to learn from the new state and reward pair
            agent.learn(self.time_step, self.restart)

            # Output a list of rewards for the step
            self.rewards[agent.agent_id] = agent.reward

        return self.global_state, self.rewards, self.restart, self.episode_info

    # Create the geometry of the desired map type
    def create_map(self, map_type):
        """
            *** DEFINING THE WALL CELLS OF THE WORLD ***
                'walls' is defined as a list with entries in the following syntax:
                    (x, y) where
                        x = x_coordinate of cell
                        y = y_coordinate of cell

            *** DEFINING THE GOAL CELLS OF THE WORLD ***
                'goals' is defined as a list with entries in the following syntax:
                    ((x, y), g_id) where
                        x = x_coordinate of cell
                        y = y_coordinate of cell
                        g_id = goal id in one-hot coding such as (0, 0, 1, 0)

            *** DEFINING THE START CELLS OF THE WORLD ***
                'starts' is defined as a list with entries in the following syntax:
                    (x, y) where
                        x = x_coordinate of cell
                        y = y_coordinate of cell
        """

        if map_type == "plus":

            width = 5
            height = 5

            walls = [(0, 0), (1, 0),     (3, 0), (4, 0),
                     (0, 1), (1, 1),     (3, 1), (4, 1),

                     (0, 3), (1, 3),     (3, 3), (4, 3),
                     (0, 4), (1, 4),     (3, 4), (4, 4)]

            goals = [((4, 2), (0, 0, 0, 1)),
                     ((2, 0), (0, 0, 1, 0)),
                     ((0, 2), (0, 1, 0, 0)),
                     ((2, 4), (1, 0, 0, 0))]

            starts = [(2, 4), (0, 2), (2, 0), (4, 2)]

            map_mode = "non-episodic"

        else:
            width = 0
            height = 0
            walls = []
            goals = []
            starts = []
            map_mode = ""

        return width, height, walls, goals, starts, map_mode

    # Create instances of agents and save in a global agent list
    def make_agents(self, agent_type):

        agent_list = []

        if agent_type is "Q_Table":

            if self.load is "yes":
                Q = "load"

            else:
                # Create a new Q-table, with all Q-values initialised to 0.1
                Q = {}
                for state in self.states:
                    temp = {}
                    for action in self.actions:
                        temp[action] = 0.1
                    Q[state] = temp  # Initialise Q table

            for i in range(self.num_agents):
                agent = Q_Table(agent_id=i, discount=0.3, epsilon_decay=0.9, actions=self.actions, q=Q)
                agent_list.append(agent)

        elif agent_type is "DQN":

            if self.load is "yes":
                Q = "load"

            else:
                Q = "new"

            for i in range(self.num_agents):
                agent = DQN(agent_id=i, discount=0.3, epsilon_decay=0.9, actions=self.actions, num_states=self.num_states, q=Q)
                agent_list.append(agent)

        elif False:  # Insert future learning methods here
            pass

        return agent_list

    # Spawn an agent in a vacant starting cell
    def spawn(self, agent):
        vacant_array = [1] * len(self.starts)
        for start in range(len(self.starts)):
            for Agent in self.agent_list:
                if Agent.position == self.starts[start]:
                    vacant_array[start] = 0
            if vacant_array[start] == 1:
                agent.position = self.starts[start]
                break

    # Create a global state by appending all agents states
    def update_global_state(self):
        self.global_state = []
        for Agent in self.agent_list:
            self.global_state.append(Agent.position[0])
            self.global_state.append(Agent.position[1])

    # Check to see if any agents have the same coordinates, indicating that a collision has occurred
    def has_collided(self, agent):

        # Remove the caller agent's own coordinates from the state list in order to compare with other agents
        other_agents = tuple(self.global_state)
        other_agents = list(other_agents)
        del other_agents[2*agent.agent_id]
        del other_agents[2*agent.agent_id]

        # Search the modified state list for other agents who occupy the same space
        for i in range(self.num_agents - 1):
            if (other_agents[2*i], other_agents[2*i+1]) == agent.position:
                return True
        return False

    # Once a goal is reached, pick a new goal for the agent
    def new_goal(self, agent):
        random_new_goal = (0, 0, 0, 0)

        # Delete the current goal from a COPY of self.goals
        copy_goals = list(self.goals)
        for ((x, y), g_id) in copy_goals:
            if agent.goal == g_id:
                copy_goals.remove(((x, y), g_id))

        (x, y) = agent.position
        while (x, y) == agent.position:
            n = randint(0, len(copy_goals) - 1)  # The next random goal is chosen here
            ((x, y), random_new_goal) = copy_goals[n]

        agent.goal = random_new_goal

    # Reformat the global state for an individual agent
    def reformat_state(self, agent, coords_type):

        reformatted_state = list(self.global_state)
        (px, py) = agent.position

        # Encode relative goal of the agent
        for ((gx, gy), g_id) in self.goals:
            if agent.goal == g_id:
                reformatted_state.append(gx)
                reformatted_state.append(gy)

        if coords_type is "relative":

            # Subtract an agent's own position from the global state
            reformatted_state[::2] = np.array(reformatted_state[::2]) - px
            reformatted_state[1::2] = np.array(reformatted_state[1::2]) - py

        elif coords_type is "absolute":
            pass

        return tuple(reformatted_state)

    # Updates an agent's intent info (in words)
    def update_intent(self, agent):
        if agent.goal == (1, 0, 0, 0):
            agent.intent = "S"
        elif agent.goal == (0, 1, 0, 0):
            agent.intent = "W"
        elif agent.goal == (0, 0, 1, 0):
            agent.intent = "N"
        elif agent.goal == (0, 0, 0, 1):
            agent.intent = "E"
        else:
            agent.intent = "None"

    # Save agents' Q-tables or Neural Networks to file
    def write_to_file(self):
        for agent in self.agent_list:
            agent.save()

    # In testing mode only
    def epsilon_greedy(self):
        for Agent in self.agent_list:
            Agent.epsilon = 0
