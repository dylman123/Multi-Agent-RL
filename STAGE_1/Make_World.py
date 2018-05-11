__author__ = 'Dylan Klein'
'''This file defines the class for creating grid world environments as an object'''

from STAGE_1.State_Space import make_states
from tkinter import *  # For rendering


class Grid_World:

    def __init__(self, width, height, initial_state, walls, specials):

        # User defined variables
        self.width = width
        self.height = height
        self.init_state = initial_state
        self.walls = walls
        self.specials = specials

        # Multi-agent variables
        self.num_agents = int(len(initial_state) / 2)  # The number of agents in this environment
        self.state_space = make_states(self.num_agents, self.width, self.height)  # Create the entire state space
        self.goal_count = 0  # To count how many agents have reached their respective goals in each episode
        self.goal_flags = [0] * self.num_agents  # To track which agents have reached their goals in each episode
        self.common_goal_flag = 0  # If agents collide on a common goal, do not punish
        self.collisions = 0  # To count how many collisions have occurred in each episode
        self.large_reward = 50  # To give if all agents collaborate
        self.crash_punishment = 10  # To punish if any agents crash

        # RL variables
        self.state = list(self.init_state)  # Set the current state to the initial state
        self.reward = 0
        self.restart = False  # Flag to indicate whether episode has restarted or not
        self.episode_count = 0  # Integer to store how many episodes have occurred

        # Environment variables
        self.actions = ["up", "down", "left", "right", "none"]
        self.wall_punishment = 1
        self.walk_punishment = 0.04

        # Create a board to render to
        self.master = Tk()
        self.L = 100  # side length of a cell in pixels
        self.board = Canvas(self.master, width=self.width * self.L, height=self.height * self.L)
        self.board.grid(row=0, column=0)

        # Render the static components of the environment
        for i in range(width):
            for j in range(height):
                self.board.create_rectangle(i * self.L, j * self.L, (i + 1) * self.L, (j + 1) * self.L, fill="white", width=1)
        for (a_id, (i, j), c, r) in specials:
            self.board.create_rectangle(i * self.L, j * self.L, (i + 1) * self.L, (j + 1) * self.L, fill=c, width=1)
        for (i, j) in walls:
            self.board.create_rectangle(i * self.L, j * self.L, (i + 1) * self.L, (j + 1) * self.L, fill="black", width=1)

        # Create objects for rendering agents
        self.objects = {}
        self.colours = ["orange", "blue", "pink", "yellow", "purple"]
        self.names = ["A", "B", "C", "D", "E"]
        for i in range(self.num_agents):
            position = (self.state[2*i], self.state[2*i+1])
            self.objects[i] = self.board.create_rectangle(position[0] * self.L + self.L * 2 / 10,
                                                          position[1] * self.L + self.L * 2 / 10,
                                                          position[0] * self.L + self.L * 8 / 10,
                                                          position[1] * self.L + self.L * 8 / 10,
                                                          fill=self.colours[i], width=1, tag=self.names[i])

    # Start rendering the 'master' object
    def start_game(self):
        self.master.mainloop()

    def reset(self):
        self.state = list(self.init_state)  # Set the current state to the initial state
        for i in range(self.num_agents):
            self.goal_flags[i] = 0
        self.goal_count = 0
        self.common_goal_flag = 0
        self.collisions = 0
        self.restart = False
        self.episode_count += 1
        return tuple(self.state)

    def render(self):
        for i in range(self.num_agents):
            position = (self.state[2*i], self.state[2*i+1])
            self.board.coords(self.objects[i],
                              position[0] * self.L + self.L * 2 / 10,
                              position[1] * self.L + self.L * 2 / 10,
                              position[0] * self.L + self.L * 8 / 10,
                              position[1] * self.L + self.L * 8 / 10)

    def step(self, i, action):

        # Initialise the reward for the step
        self.reward = 0

        # Calculate incremental changes in x,y coordinates
        if action == self.actions[0]:
            (dx, dy) = (0, -1)  # UP
        elif action == self.actions[1]:
            (dx, dy) = (0, 1)  # DOWN
        elif action == self.actions[2]:
            (dx, dy) = (-1, 0)  # LEFT
        elif action == self.actions[3]:
            (dx, dy) = (1, 0)  # RIGHT
        else:
            (dx, dy) = (0, 0)  # NONE

        # Calculate the new proposed position
        position = (self.state[2*i], self.state[2*i+1])
        new_x = position[0] + dx
        new_y = position[1] + dy

        # Try move the caller agent to a new cell if possible
        if (new_x >= 0) and (new_x < self.width) and (new_y >= 0) and (new_y < self.height) and not ((new_x, new_y) in self.walls):
            (self.state[2*i], self.state[2*i+1]) = (new_x, new_y)
            self.common_goal_flag = 0
            self.reward = -self.walk_punishment
        else:
            self.reward = -self.wall_punishment  # Punish if a wall is touched (for faster convergence)

        # Check for landing on a green or red space
        for (a_id, (x, y), c, r) in self.specials:
            if new_x == x and new_y == y and (a_id == i+1 or a_id == 0):
                self.reward = r
                if a_id == 0:
                    self.common_goal_flag = 1
                if c == "green":
                    if self.goal_flags[i] == 0:
                        self.goal_flags[i] = 1
                        self.goal_count = sum(tuple(self.goal_flags))
                elif c == "red":
                    pass

        # Only if all agents are at their respective goals, restart the game
        if self.goal_count == self.num_agents:
            self.reward = self.large_reward  # If all agents reach their goals, give a large reward to all
            self.restart = True

        # Check for a collision of agents
        for n in range(self.num_agents):
            if Grid_World.has_collided(self, i) is True and self.common_goal_flag == 0:
                for j in range(self.num_agents):
                    self.reward = -self.crash_punishment
                self.restart = True
                self.collisions += 1

        return tuple(self.state), self.reward, self.restart, self.episode_count

    # Check to see if any agents have the same coordinates, indicating that a collision has occurred
    def has_collided(self, i):

        # Remove the caller agent's own coordinates from the state list in order to compare with other agents
        other_agents = tuple(self.state)
        other_agents = list(other_agents)
        del other_agents[2*i]
        del other_agents[2*i]

        # Search the modified state list for other agents who occupy the same space
        for j in range(self.num_agents - 1):
            if (other_agents[2*j], other_agents[2*j+1]) == (self.state[2*i], self.state[2*i+1]):
                return True
            else:
                pass
        return False
