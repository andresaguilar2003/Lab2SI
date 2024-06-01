import numpy as np
import random
import json
import time
from draw_policy import draw_policy_map

class Environment:
    def __init__(self, config, transition_prob=1.0):
        self.rows = config['city']['rows']
        self.columns = config['city']['columns']
        self.grid = np.full((self.rows, self.columns), ' ')
        self.start_state = tuple(config['departure'])
        self.transition_prob = transition_prob

        self.blocked = set(map(tuple, config['city']['blocked']))
        self.dangers = set(map(tuple, config['dangers']))
        self.trapped = {tuple(pos[:2]): pos[2] for pos in config['trapped']}
        self.fatal_dangers = {tuple(pos[:2]): pos[2] for pos in config['fatal_dangers']}

        for b in self.blocked:
            self.grid[b] = 'B'
        for d in self.dangers:
            self.grid[d] = 'A'
        for t in self.trapped:
            self.grid[t] = 'T'
        for f in self.fatal_dangers:
            self.grid[f] = 'F'

    def get_next_state(self, state, action):
        row, col = state
        intended = state
        perpendiculars = [(row, col - 1), (row, col + 1), (row - 1, col), (row + 1, col)]
        if action == 'Up':
            intended = (row - 1, col)
        elif action == 'Down':
            intended = (row + 1, col)
        elif action == 'Left':
            intended = (row, col - 1)
        elif action == 'Right':
            intended = (row, col + 1)

        perpendiculars = [s for s in perpendiculars if self.is_valid_state(s) and s != intended]

        if not perpendiculars:
            perpendiculars = [state]

        if np.random.rand() < self.transition_prob:
            next_state = intended
        else:
            next_state = random.choice(perpendiculars)

        if not self.is_valid_state(next_state):
            next_state = state

        return next_state

    def is_valid_state(self, state):
        row, col = state
        if row < 0 or row >= self.rows or col < 0 or col >= self.columns:
            return False
        if (row, col) in self.blocked:
            return False
        return True

    def get_reward(self, state):
        if state in self.trapped:
            return self.trapped[state]
        elif state in self.dangers:
            return -5.0
        elif state in self.fatal_dangers:
            return self.fatal_dangers[state]
        return -0.1

    def is_terminal_state(self, state):
        return state in self.trapped or state in self.fatal_dangers


class Q_Learning_Agent:
    def __init__(self, environment, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.environment = environment
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros((environment.rows, environment.columns, 4))
        self.actions = ['Up', 'Down', 'Left', 'Right']
        self.action_map = {a: i for i, a in enumerate(self.actions)}

    def get_valid_actions(self, state):
        row, col = state
        valid_actions = []
        if row > 0:
            valid_actions.append('Up')
        if row < self.environment.rows - 1:
            valid_actions.append('Down')
        if col > 0:
            valid_actions.append('Left')
        if col < self.environment.columns - 1:
            valid_actions.append('Right')
        return valid_actions

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.get_valid_actions(state))
        else:
            valid_actions = self.get_valid_actions(state)
            row, col = state
            valid_actions_values = np.array([])
            for a in valid_actions:
                valid_actions_values = np.append(valid_actions_values, self.q_table[row, col, self.action_map[a]])
            return valid_actions[np.argmax(valid_actions_values)]

    def update_q_value(self, state, action, reward, next_state):
        row, col = state
        next_row, next_col = next_state
        action_idx = self.action_map[action]
        old_value = self.q_table[row, col, action_idx]
        next_max = np.max(self.q_table[next_row, next_col])
        self.q_table[row, col, action_idx] = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

    def train(self, episodes):
        for episode in range(episodes):
            state = self.environment.start_state
            done = False
            while not done:
                action = self.choose_action(state)
                next_state = self.environment.get_next_state(state, action)
                reward = self.environment.get_reward(next_state)
                self.update_q_value(state, action, reward, next_state)
                if self.environment.is_terminal_state(next_state):
                    done = True
                state = next_state
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{episodes} completed. Epsilon: {self.epsilon:.4f}")
        print("Training completed.")
        print("Q-table:")
        print(self.q_table)

    def get_policy(self):
        policy = {}
        for row in range(self.environment.rows):
            for col in range(self.environment.columns):
                if self.environment.grid[row, col] not in ['B', 'T', 'F']:
                    state = (row, col)
                    valid_actions = self.get_valid_actions(state)
                    valid_actions_values = np.array([])
                    for a in valid_actions:
                        valid_actions_values = np.append(valid_actions_values,
                                                         self.q_table[row, col, self.action_map[a]])
                    policy[row, col] = valid_actions[np.argmax(valid_actions_values)]
        return policy


class Policy_Iteration_Agent:
    def __init__(self, environment, gamma=0.9, theta=1e-3):
        self.environment = environment
        self.gamma = gamma
        self.theta = theta
        self.actions = ['Up', 'Down', 'Left', 'Right']
        self.policy = np.empty((environment.rows, environment.columns), dtype='<U5')
        self.value_function = np.zeros((environment.rows, environment.columns))

        for row in range(environment.rows):
            for col in range(environment.columns):
                state = (row, col)
                if state not in environment.blocked and not environment.is_terminal_state(state):
                    self.policy[row, col] = random.choice(self.get_valid_actions(state))
                else:
                    self.policy[row, col] = ' '

    def get_valid_actions(self, state):
        row, col = state
        valid_actions = []
        if row > 0:
            valid_actions.append('Up')
        if row < self.environment.rows - 1:
            valid_actions.append('Down')
        if col > 0:
            valid_actions.append('Left')
        if col < self.environment.columns - 1:
            valid_actions.append('Right')
        return valid_actions

    def policy_evaluation(self):
        while True:
            delta = 0
            for row in range(self.environment.rows):
                for col in range(self.environment.columns):
                    state = (row, col)
                    if self.environment.is_terminal_state(state) or state in self.environment.blocked:
                        continue
                    v = self.value_function[row, col]
                    action = self.policy[row, col]
                    next_state = self.environment.get_next_state(state, action)
                    reward = self.environment.get_reward(next_state)
                    self.value_function[row, col] = reward + self.gamma * self.value_function[next_state]
                    delta = max(delta, abs(v - self.value_function[row, col]))
            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for row in range(self.environment.rows):
            for col in range(self.environment.columns):
                state = (row, col)
                if self.environment.is_terminal_state(state) or state in self.environment.blocked:
                    continue
                old_action = self.policy[row, col]
                action_values = {}
                for action in self.get_valid_actions(state):
                    next_state = self.environment.get_next_state(state, action)
                    reward = self.environment.get_reward(next_state)
                    action_values[action] = reward + self.gamma * self.value_function[next_state]
                new_action = max(action_values, key=action_values.get)
                self.policy[row, col] = new_action
                if new_action != old_action:
                    policy_stable = False
        return policy_stable

    def policy_iteration(self):
        iteration = 0
        max_iterations = 100  # o cualquier número razonable
        while iteration < max_iterations:
            print(f"Policy Iteration step {iteration}")
            self.policy_evaluation()
            if self.policy_improvement():
                break
            iteration += 1
        if iteration == max_iterations:
            print("Reached maximum iterations")
        print("Policy Iteration completed.")
        print("Policy:")
        print(self.policy)
        print("Value function:")
        print(self.value_function)


# Parámetros de experimentación
maze_sizes = [(1, 7), (10, 10)]
seeds = [42, 259, 1020]
transition_probs = [1.0, 0.8, 0.4]
gamma_values = [1.0, 0.95]
episodes = 50


# Función para cargar configuraciones del laberinto
def load_maze_config(rows, cols):
    config_path = f"tests/easy/maze_{rows}x{cols}.json"  # Ajusta el path según tu estructura
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config


# Función para ejecutar un experimento
def run_experiment(maze_size, seed, transition_prob, gamma, episodes):
    print(f"Loading config for maze size {maze_size}")
    rows, cols = maze_size
    np.random.seed(seed)
    random.seed(seed)

    config = load_maze_config(rows, cols)
    environment = Environment(config, transition_prob=transition_prob)

    # Política Iterativa
    #print("Starting Policy Iteration")
    #pi_agent = Policy_Iteration_Agent(environment, gamma=gamma)
    #start_time = time.time()
    #pi_agent.policy_iteration()
    #pi_time = time.time() - start_time
    #pi_iterations = pi_agent.value_function.size

    # Q-Learning
    print("Starting Q-Learning")
    q_agent = Q_Learning_Agent(environment, gamma=gamma)
    start_time = time.time()
    q_agent.train(episodes=episodes)
    q_time = time.time() - start_time

    #pi_policy = pi_agent.policy
    q_policy = q_agent.get_policy()

    #pi_initial_utility = pi_agent.value_function[environment.start_state]
    q_initial_utility = q_agent.q_table[environment.start_state][np.argmax(q_agent.q_table[environment.start_state])]


    return {
        "maze_size": maze_size,
        "seed": seed,
        "transition_prob": transition_prob,
        "gamma": gamma,
        #"pi_iterations": pi_iterations,
        #"pi_time": pi_time,
        #"pi_initial_utility": pi_initial_utility,
        "q_time": q_time,
        "q_initial_utility": q_initial_utility,
        "q_policy": q_policy,
    }


# Función principal para ejecutar todos los experimentos
def main():
    results = []
    for maze_size in maze_sizes:
        for seed in seeds:
            for transition_prob in transition_probs:
                for gamma in gamma_values:
                    print(
                        f"Running experiment with size {maze_size}, seed {seed}, transition_prob {transition_prob}, gamma {gamma}")
                    result = run_experiment(maze_size, seed, transition_prob, gamma, episodes)
                    results.append(result)
                    print(f"Experiment completed: {result}")


if __name__ == "__main__":
    main()
