from pettingzoo.test import api_test

import src.environment.Env as Env
from src.environment.Config import Config

def run_policy(policy):
	config = Config()
	env = Env.env(config)
	api_test(env, num_cycles=10, verbose_progress=False)

	env.reset()
	iter_num = -1
	lifespans = {agent: 0 for agent in env.agents}
	for agent in env.agent_iter():
		observation, reward, done, info = env.last() # note that info is empty
		if not done:
			lifespans[agent] += 1
		action = policy(observation, agent) if not done else None
		env.step(action)
		env.render()
	print("lifespans", lifespans)
	print("average lifespan", sum(lifespans.values()) / len(lifespans))
