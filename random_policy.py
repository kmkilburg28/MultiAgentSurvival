from random import randint
from pettingzoo.test import api_test

import Env
from Config import Config
from Direction import Direction
from Tasks import Tasks

# Random policy
max_dir = max([direction.value for direction in list(Direction)])
max_task = max([task.value for task in list(Tasks)])
def policy(observation, agent):
	return [randint(0,1),randint(0,max_dir),randint(0,max_task)]

if __name__ == "__main__":
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
