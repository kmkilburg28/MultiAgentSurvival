from gym import spaces
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
import torch
from torch import nn
import tensorflow as tf
from ray.rllib.agents.ppo import PPOTrainer
import supersuit as ss
import numpy as np

import Env
from Config import Config

def env_creator():
	config = Config()
	env = Env.env(config)
	# env = ss.frame_stack_v1(env, 3)
	def change_action_fn(action, old_space, agent=None):
		action = np.maximum(old_space.low, action)
		action = np.minimum(old_space.high, action)
		return action
	def change_space_fn(old_space, agent=None):
		low_ = np.min(old_space.low)
		high_ = np.max(old_space.high)
		return spaces.Box(low=low_, high=high_, shape=old_space._shape, dtype=old_space.dtype)
	env = ss.action_lambda_v1(env, change_action_fn, change_space_fn)
	return env

env_name = "MultiAgentSurvival_v0"
env = env_creator()
register_env(env_name, lambda config: PettingZooEnv(env))

# Create an RLlib Trainer instance.
trainer = PPOTrainer(
    config={
        # Env class to use (here: our gym.Env sub-class from above).
        "env": env_name,
        # Config dict to be passed to our custom env's constructor.
        "env_config": {
            # Use corridor with 20 fields (including S and G).
            "corridor_length": 20
        },
        # Parallelize environment rollouts.
        "num_workers": 3,
    })

# Train for n iterations and report results (mean episode rewards).
# Since we have to move at least 19 times in the env to reach the goal and
# each move gives us -0.1 reward (except the last move at the end: +1.0),
# we can expect to reach an optimal episode reward of -0.1*18 + 1.0 = -0.8
env.reset()
print("Beginning training")
for i in range(100):
    results = trainer.train()
    print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")

# Perform inference (action computations) based on given env observations.
# Note that we are using a slightly different env here (len 10 instead of 20),
# however, this should still work as the agent has (hopefully) learned
# to "just always walk right!"

env.reset()
for agent in env.agent_iter():
	observation, reward, done, info = env.last() # note that info is empty
	print(reward, done)
	# trainer.
	action = trainer.compute_single_action(observation) if not done else None
	env.step(action)
	env.render()

# # Get the initial observation (should be: [0.0] for the starting position).
# obs = env.reset()
# done = False
# total_reward = 0.0
# # Play one episode.
# while not done:
#     # Compute a single action, given the current observation
#     # from the environment.
#     action = trainer.compute_single_action(obs)
#     # Apply the computed action in the environment.
#     obs, reward, done, info = env.step(action)
#     # Sum up rewards for reporting purposes.
#     total_reward += reward
# # Report results.
