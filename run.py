from Config import Config
from Direction import Direction
import Env
from pettingzoo.test import parallel_api_test, api_test

config = Config()
env = Env.env(config)
api_test(env, num_cycles=10, verbose_progress=False)

def policy(observation, agent):
	return [1,Direction.WEST.value,0]

env.reset()
for agent in env.agent_iter():
	observation, reward, done, info = env.last() # note that info is empty
	print(reward, done)
	action = policy(observation, agent) if not done else None
	env.step(action)
	env.render()
