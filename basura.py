import json
import numpy as np
import random

#Tengo que hacer que el agente no elija una opcion no valida, sease ni bloqueada ni out of bound, por lo que una
#opción podría ser hacer state una clase que tenga (x, y) y acciones posibles, y lo primero que se haga se calcular
#las opciones posibles del estado actual

class state:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.possible_actions = []

    def set_possible_actions(self, possible_actions):
        self.possible_actions = possible_actions

class UrbanRescueEnvironment:

    def __init__(self, city, departure, dangers, trapped, fatal_dangers, stochasticity):
        self.blocked = np.array(city["blocked"])
        self.departure = np.array(departure)
        self.dangers = np.array(dangers)
        self.trapped = np.array(trapped)
        self.fatal_dangers = np.array(fatal_dangers)
        self.stochasticity = stochasticity

        self.rows = city["rows"]
        self.columns = city["columns"]

        self.current_state = state(self.departure[0], self.departure[1])

        self.is_terminal = False

    def take_action(self, action):
        if random.random() > self.stochasticity:
            possible_actions = self.get_possible_actions(self.current_state)
            action = random.choice(possible_actions)

        new_state = self.move(action)

        new_state.set_possible_actions(self.get_possible_actions(new_state))

        reward = self.calculate_reward(new_state)

        new_state_coord = np.array([new_state.x, new_state.y])
        if len(self.trapped) >0:
            if np.array_equal(new_state_coord, self.trapped):
                self.is_terminal = True

        if len(self.fatal_dangers) > 0:
            if np.array_equal(new_state_coord, self.fatal_dangers):
                self.is_terminal = True

        self.current_state = new_state

        return new_state, reward, self.is_terminal

    def get_possible_actions(self, _state):

        possible_actions = ["UP", "RIGHT", "DOWN", "LEFT"]
        x, y = _state.x, _state.y
        if x - 1 < 0:
            possible_actions.remove("UP")
        if x + 1 >= self.rows:
            possible_actions.remove("DOWN")
        if y - 1 < 0:
            possible_actions.remove("LEFT")
        if y + 1 >= self.columns:
            possible_actions.remove("RIGHT")

        if len(self.blocked)>0:
            if np.any(np.all(self.blocked == [x - 1, y], axis=1)):
                possible_actions.remove("UP")
            if np.any(np.all(self.blocked == [x + 1, y], axis=1)):
                possible_actions.remove("DOWN")
            if np.any(np.all(self.blocked == [x, y - 1], axis=1)):
                possible_actions.remove("LEFT")
            if np.any(np.all(self.blocked == [x, y + 1], axis=1)):
                possible_actions.remove("RIGHT")

        return possible_actions

    def move(self, action):
        x, y = self.current_state.x, self.current_state.y

        if action == "UP":
            new_state = state(x-1, y)
        elif action == "RIGHT":
            new_state = state(x, y + 1)
        elif action == "DOWN":
            new_state = state(x + 1, y)
        elif action == "LEFT":
            new_state = state(x, y - 1)

        return new_state

    def calculate_reward(self, new_state):

        new_state_coord = (new_state.x, new_state.y)

        # Verifica si self.dangers no está vacío antes de realizar la operación
        if len(self.dangers) > 0:
            if np.any(np.all(new_state_coord == self.dangers, axis=1)):
                return -5.0

        # Verifica si self.trapped no está vacío antes de realizar la operación
        if len(self.trapped) > 0:
            if np.any(np.all(new_state_coord == self.trapped[:, :2], axis=1)):
                reward_index = np.where(np.all(self.trapped[:, :2] == new_state_coord, axis=1))[0][0]
                return self.trapped[reward_index, 2]

        # Verifica si self.fatal_dangers no está vacío antes de realizar la operación
        if len(self.fatal_dangers) > 0:
            if np.any(np.all(new_state_coord == self.fatal_dangers[:, :2], axis=1)):
                reward_index = np.where(np.all(self.fatal_dangers[:, :2] == new_state_coord, axis=1))[0][0]
                return self.fatal_dangers[reward_index, 2]

        # Si ninguna de las condiciones anteriores se cumple, devuelve 0.0
        return 0.0


class QLearningAgent:
    def __init__(self, alpha, gamma, epsilon, actions):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration-exploitation trade-off
        self.actions = actions
        self.q_table = {}

    def choose_action(self, current_state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return self.get_best_action(current_state)

    def get_best_action(self, current_state):
        current_state_q = (current_state.x, current_state.y)  # Convert the state to a tuple

        if current_state_q not in self.q_table:
            self.q_table[current_state_q] = {action: 0.0 for action in self.actions}

        return max(self.q_table[current_state_q], key=self.q_table[current_state_q].get)

    def update_q_table(self, current_state, action, reward, new_state):
        current_state_coord = (current_state.x, current_state.y) # Convert the state to a tuple
        new_state_coord = (new_state.x, new_state.y)  # Convert the new state to a tuple

        if current_state_coord not in self.q_table:
            self.q_table[current_state_coord] = {action: 0.0 for action in self.actions}

        if new_state_coord not in self.q_table:
            self.q_table[new_state_coord] = {action: 0.0 for action in self.actions}

        best_future_action = self.get_best_action(new_state)
        self.q_table[current_state_coord][action] += self.alpha * (reward + self.gamma * self.q_table[new_state_coord][best_future_action] - self.q_table[current_state_coord][action])

    def extract_policy(self):
        policy = {}
        for state in self.q_table:
            best_action = self.get_best_action(state)
            policy[state] = best_action

        return policy

def load_problem_instance(file_path):
    with open(file_path, 'r') as file:
        problem_instance = json.load(file)

    return problem_instance

def main():
    # Load problem instance from file
    file_path = '../../Desktop/Lab2_Andres_Aguilar_y_Alberto_Bueno/tests/easy/maze_1x7.json'
    problem_instance = load_problem_instance(file_path)

    print(problem_instance)

    # Extract parameters from the problem instance
    city = problem_instance["city"]
    departure = problem_instance["departure"]
    dangers = problem_instance["dangers"]
    trapped = problem_instance["trapped"]
    fatal_dangers = problem_instance["fatal_dangers"]

    # Create the environment
    stochasticity = 0.6  # Adjust as needed
    environment = UrbanRescueEnvironment(city, departure, dangers, trapped, fatal_dangers, stochasticity)

    # Create the Q-learning agent
    alpha = 0.1  # Adjust as needed
    gamma = 0.9  # Adjust as needed
    epsilon = 0.1  # Adjust as needed
    actions = ["UP", "RIGHT", "DOWN", "LEFT"]
    agent = QLearningAgent(alpha, gamma, epsilon, actions)

    # Training the agent
    num_episodes = 1  # Adjust as needed
    for _ in range(num_episodes):
        current_state = state(environment.departure[0], environment.departure[1])
        is_terminal = False

        while not is_terminal:
            possible_actions = environment.get_possible_actions(current_state)
            current_state.set_possible_actions(possible_actions)
            action = agent.choose_action(current_state)
            new_state, reward, is_terminal = environment.take_action(action)
            agent.update_q_table(current_state, action, reward, new_state)
            current_state = new_state

    # Extract the learned policy
    learned_policy = agent.extract_policy()
    print("Learned Policy:")
    print(learned_policy)


if __name__ == "__main__":
    main()