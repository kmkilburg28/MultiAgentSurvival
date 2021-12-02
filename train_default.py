from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.agents.ppo import PPOTrainer

import Env
from Config import Config

def env_creator():
	config = Config()
	env = Env.env(config)
	return env

env_name = "MultiAgentSurvival_v0"
env = env_creator()
register_env(env_name, lambda config: PettingZooEnv(env))

# Create an RLlib Trainer instance.
trainer = PPOTrainer(
	config={
		# Env class to use (here: our gym.Env sub-class from above).
		"env": env_name,

		# Parallelize environment rollouts.
		"num_workers": 3,
	}
)

# Train for n iterations
# trainer.load_checkpoint("checkpoint/checkpoint_000100/checkpoint-100")
env.reset()
print("Beginning training")
for i in range(100):
	results = trainer.train()
	print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")
checkpoint_path = trainer.save("checkpoint")
print(checkpoint_path)

# Test the trained model
# trainer.load_checkpoint("checkpoint/checkpoint_000100/checkpoint-100")
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
