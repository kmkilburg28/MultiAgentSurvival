"""Note: This script runs but fails to produce logical agents"""

from ray import tune
import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
import torch
from torch import nn
import tensorflow as tf
import numpy as np
from gym import spaces
from supersuit import action_lambda_v1

import Env
from Config import Config

class MyModelClass(TFModelV2): #, nn.Module):
	def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
		super(MyModelClass, self).__init__(obs_space, act_space, num_outputs, *args, **kwargs)

		grid_inputs = [
			tf.keras.layers.Input(shape=obs_space.original_space['grid']._shape),
			tf.keras.layers.Input(shape=obs_space.original_space['dropped']._shape),
			tf.keras.layers.Input(shape=obs_space.original_space['agents']._shape),
		]
		status_input = tf.keras.layers.Input(shape=obs_space.original_space['status']._shape)
		inputs = grid_inputs + [status_input]

		layer = tf.keras.layers.Concatenate()(grid_inputs)
		layer = tf.keras.layers.Flatten()(layer)
		layer = tf.keras.layers.Concatenate()([layer, status_input])
  
		self.max_action_space = max(self.action_space.nvec)
		policy_outputs = tf.keras.layers.Concatenate()([
			tf.keras.layers.Dense(units=size, activation='softmax')(layer) for size in self.action_space.nvec
			# tf.keras.layers.Dense(units=self.max_action_space)(layer) for size in self.action_space.nvec
			# tf.keras.layers.Dense(units=size, activation='softmax')(layer) for size in self.action_space.nvec
		])
		value_outputs = tf.keras.layers.Dense(units=1)(layer)
		self.model = tf.keras.models.Model(inputs, [policy_outputs, value_outputs])
		print(self.model)


		self.register_variables(self.model.variables)
		# self.model = nn.Sequential(
		# 	nn.Conv2d( 3, 32, [8, 8], stride=(4, 4)),
		# 	nn.ReLU(),
		# 	nn.Conv2d( 32, 64, [4, 4], stride=(2, 2)),
		# 	nn.ReLU(),
		# 	nn.Conv2d( 64, 64, [3, 3], stride=(1, 1)),
		# 	nn.ReLU(),
		# 	nn.Flatten(),
		# 	(nn.Linear(3136,512)),
		# 	nn.ReLU(),
		# )
		# self.policy_fn = nn.Linear(512, num_outputs)
		# self.value_fn = nn.Linear(512, 1)
	def forward(self, input_dict, state, seq_lens):

		# print("PREDICTING!")
		inputs = [
			input_dict['obs']['grid'],
			input_dict['obs']['dropped'],
			input_dict['obs']['agents'],
			input_dict['obs']['status']
		]

		# Logits- and value branches.
		logits, values = self.model(inputs)
		self._value_out = tf.reshape(values, [-1])
		return logits, []

		return policy_out, []
		# model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
		# self._value_out = self.value_fn(model_out)
		# return self.policy_fn(model_out), state
	def value_function(self):
		# return tf.constant([0.0])
		# print("VALUE_OUT", self._value_out)
		return self._value_out

def env_creator():
	config = Config()
	env = Env.env(config)
	# def change_action_fn(action, space):
	# 	print("CHANGING ACTION!", action)
	# 	action = [min(max(0, action[i]), space.nvec[i]) for i in range(len(action))]
	# 	print(action)
	# 	return action
	# def change_space_fn(action_space):
	# 	print("CHANGING SPACE", action_space)
	# 	max_action = max(action_space.nvec)
	# 	action_space = spaces.MultiDiscrete([max_action for action in action_space.nvec])
	# 	print(action_space)
	# 	return action_space
	# env = action_lambda_v1(env, change_action_fn, change_space_fn)
	# env = ss.frame_stack_v1(env, 3)
	return env

if __name__ == "__main__":
	env_name = "MultiAgentSurvival_v0"
	env = PettingZooEnv(env_creator())
	register_env(env_name, lambda config: env)
	obs_space = env.observation_space
	act_space = env.action_space

	ModelCatalog.register_custom_model("my_tf_model", MyModelClass)
	ray.init()
	trainer = ppo.PPOTrainer(env="MultiAgentSurvival_v0", config={
		"model": {
			"custom_model": "my_tf_model",
			# Extra kwargs to be passed to your model's c'tor.
			"custom_model_config": {},
		},
	})

	# Train for n iterations
	trainer.load_checkpoint("checkpoint_custom_tf/checkpoint_000200/checkpoint-200")
	# env.reset()
	# print("Beginning training")
	# for i in range(200):
	# 	results = trainer.train()
	# 	print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")
	# checkpoint_path = trainer.save("checkpoint_custom_tf")
	# print(checkpoint_path)

	# ModelCatalog.register_custom_model("my_tf_model", MyModelClass)
	# def gen_policy(i):
	# 	config = {
	# 		"model": {
	# 			"custom_model": "my_tf_model",
	# 		},
	# 			"gamma": 0.99,
	# 	}
	# 	return (None, obs_space, act_space, config)
	# policies = {agent: gen_policy(agent) for agent in env.agents}
	# policy_ids = list(policies.keys())
	# tune.run(
	# 	"PPO",
	# 	name="PPO",
	# 	stop={"timesteps_total": 5000000},
	# 	checkpoint_freq=10,
	# 	local_dir="ray_results/"+env_name,
	# 	config={
	# 		# Environment specific
	# 		"env": env_name,
	# 		# General
	# 		"log_level": "ERROR",
	# 		"framework": "tensorflow",
	# 		"num_gpus": 0,
	# 		"num_workers": 1,#3,
	# 		"num_envs_per_worker": 1,
	# 		"compress_observations": False,
	# 		"batch_mode": 'truncate_episodes',
	# 		# ‘use_critic’: True,
	# 		'use_gae': True,
	# 		"lambda": 0.9,
	# 		"gamma": .99,
	# 		# "kl_coeff": 0.001,
	# 		# "kl_target": 1000.,
	# 		"clip_param": 0.4,
	# 		'grad_clip': None,
	# 		"entropy_coeff": 0.1,
	# 		'vf_loss_coeff': 0.25,
	# 		"sgd_minibatch_size": 64,
	# 		"num_sgd_iter": 10, # epoc
	# 		'rollout_fragment_length': 512,
	# 		"train_batch_size": 512*4,
	# 		'lr': 2e-05,
	# 		"clip_actions": True,
	# 		# Method specific
	# 		"multiagent": {
	# 			"policies": policies,
	# 			"policy_mapping_fn": (
	# 			lambda agent_id: policies[agent_id]),
	# 		},
	# 	},
	# )

	env = env.env
	env.reset()
	iter_num = -1
	lifespans = {agent: 0 for agent in env.agents}
	for agent in env.agent_iter():
		observation, reward, done, info = env.last() # note that info is empty
		if not done:
			lifespans[agent] += 1
		action = trainer.compute_single_action(observation) if not done else None
		env.step(action)
		env.render()
	print("lifespans", lifespans)
	print("average lifespan", sum(lifespans.values()) / len(lifespans))