import numpy as np
import ray
from ray.rllib.agents import ppo
import datetime

class TransportScape:
    def __init__(self, config):
        # Initialize environment parameters
        self.reward = 0
        self.done = False
        self.state = {
            'truck location': [],
            'assignment': [],
            'truck usage': [],
            'time left': []
        }
        self.distances = np.array([])  # Placeholder for distances
        self.speed = 1.0  # Placeholder for speed
        # Other initializations

    def reset(self):
        # Reset environment state
        self.state = {
            'truck location': [],
            'assignment': [],
            'truck usage': [],
            'time left': []
        }
        self.reward = 0
        self.done = False
        return self.state

    def step(self, action):
        # Unpack action and update state
        truck_use = self.state['truck usage']
        truck_loc = self.state['truck location']
        assignment = self.state['assignment']
        time_left = self.state['time left']
        i, j = action

        if truck_use[i] == 0:
            self.reward -= 2000
            truck_use[i] = 1

        assignment[j] = 1
        self.reward -= self.distances[truck_loc[i], j]
        time_left[i] -= self.distances[truck_loc[i], j] / self.speed - 0.5
        truck_loc[i] = j

        self.done = np.all(assignment == 1)

        self.state['truck location'] = truck_loc
        self.state['assignment'] = assignment
        self.state['truck usage'] = truck_use
        self.state['time left'] = time_left

        return self.state, self.reward, self.done, {}

# PPO Configuration
config = ppo.DEFAULT_CONFIG.copy()
config["env_config"] = {}
config["num_workers"] = 8
config["framework"] = "torch"
config["kl_coeff"] = 0.0
config["log_level"] = "ERROR"
config["num_gpus"] = 0
config["output"] = "/home/basilshim/ray_results/ppo_n10"
config["clip_param"] = 0.3
config["entropy_coeff"] = 0.01
config["lr"] = 0.0001

# Initialize Ray and PPO Trainer
ray.shutdown()
ray.init()
agent = ppo.PPOTrainer(config=config, env=TransportScape)

# Training Loop
start = datetime.datetime.now()
print(start)
for i in range(101):
    result = agent.train()
    if i % 10 == 0:
        print(f'i: {i}')
        print(f'mean episode length: {result["episode_len_mean"]}')
        print(f'max episode reward: {result["episode_reward_max"]}')
        print(f'mean episode reward: {result["episode_reward_mean"]}')
        print(f'min episode reward: {result["episode_reward_min"]}')
        print(f'total episodes: {result["episodes_total"]}')
        checkpoint = agent.save()

finish = datetime.datetime.now()
print(finish)
duration = finish - start
print(f"Total Hours: {duration.total_seconds() / 3600}")

# Simulation
env = TransportScape(config)
state = env.reset()
g = 0
done = False
reward = 0
actions = []
i = 0

while not done:
    action = agent.compute_action(state, explore=False)
    actions.append(action)
    print(f"action: {action}; reward: {reward}")
    state, reward, done, info = env.step(action)
    g += reward
    i += 1
    if i == 1000:
        break

print("Final state:")
print(f"Assignment: {state['assignment']}")
print(f"Trucks used: {state['truck usage']}")
print(f"Time left: {state['time left']}")
print(g)

# Path Extraction
truck_paths = {}
for k, v in actions:
    truck_paths.setdefault(k, []).append(v)

for k in truck_paths:
    print(f"Truck #{k}: {truck_paths[k]}")

# Route Length Calculation
def GetLength(route, df):
    length = df.values[-1, route[0]]
    for i in range(1, len(route):
        length += df.values[route[i-1], route[i]]
    length += df.values[route[-1], -1]
    return length

# Total Distance Calculation
dist = 0
for k in truck_paths:
    dist += GetLength(truck_paths[k], df)
print(dist))
