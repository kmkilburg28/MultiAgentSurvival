from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.pg import PGTrainer
from ray.rllib.agents.marwil import BCTrainer

import sys
import os
sys.path.append(os.getcwd())

import src.environment.Env as Env
from src.environment.Config import Config
from src.models.utils.CustomMetricsCallbacks import CustomMetricsCallbacks
from src.models.utils.logger import Logger
import os


# {name: (TrainerClass, epochs, customconfig)}
trainings = {
	"bc": (BCTrainer, 2000, {}),
	"pg": (PGTrainer, 2000, {}),
	"ppo": (PPOTrainer, 300, {}),
	"ppo_lstm": (PPOTrainer, 300, {
		"model": {
			"use_lstm": True,
		},
	}),
	"ppo_attention": (PPOTrainer, 300, {
		"model": {
			"use_attention": True,
		},
	}),
}


def env_creator():
	config = Config()
	config.RENDER_ENABLE = False
	env = Env.env(config)
	return env

env_name = "MultiAgentSurvival_v0"
base_config = {
	# Env class to use (here: our gym.Env sub-class from above).
	"env": env_name,
	# Parallelize environment rollouts.
	"num_workers": 3,
	"callbacks": CustomMetricsCallbacks,
}

env = env_creator()
register_env(env_name, lambda config: PettingZooEnv(env))

for training in trainings:
	print("NEW TrainerClass", training)

	logger = Logger(training)

	# Create an RLlib Trainer instance.
	TrainerClass = trainings[training][0]
	custom_config = base_config.copy()
	custom_config.update(trainings[training][2])
	print(TrainerClass, custom_config)
	trainer = TrainerClass(
		config=custom_config
	)

	# Train for n iterations
	checkpoint_dir = os.path.join("checkpoints", training)
	env.reset()
	print("Beginning training")
	epochs = trainings[training][1]
	for i in range(epochs):
		results = trainer.train()
		logger.log(results)
		print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}, avg. lifespan={results['custom_metrics']['avg_lifespan_mean']}")
	checkpoint_path = trainer.save(checkpoint_dir)

	# # Test the trained model
	# # trainer.load_checkpoint(checkpoint_path)
	# trainer.load_checkpoint(checkpoint_path)
	# env.reset()
	# iter_num = -1
	# lifespans = {agent: 0 for agent in env.agents}
	# for agent in env.agent_iter():
	# 	observation, reward, done, info = env.last() # note that info is empty
	# 	if not done:
	# 		lifespans[agent] += 1
	# 	action = trainer.compute_single_action(observation) if not done else None
	# 	env.step(action)
	# 	env.render()
	# print("lifespans", lifespans)
	# print("average lifespan", sum(lifespans.values()) / len(lifespans))
