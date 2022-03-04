import sys
import os
sys.path.append(os.getcwd())

import src.environment.Env as Env
from src.environment.Config import Config

from src.models.static.idle_policy import policy as IdlePolicy
from src.models.static.random_policy import policy as RandomPolicy
from src.models.static.SearchPolicy import policy as SearchPolicy
from src.models.utils.logger import Logger

if __name__ == "__main__":
	policies = {
		"IdlePolicy"  : IdlePolicy,
		"RandomPolicy": RandomPolicy,
		"SearchPolicy": SearchPolicy,
	}

	batch_size = 64


	for policyName in policies:
		policy = policies[policyName]
		config = Config()
		config.RENDER_ENABLE = False
		env = Env.env(config)

		avg_lifespan_sum = 0
		max_lifespan_sum = 0
		max_lifespan_max = 0
		for batch in range(batch_size):
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
			avg_lifespan = sum(lifespans.values()) / len(lifespans)
			max_lifespan = max(lifespans.values())
			avg_lifespan_sum += avg_lifespan
			max_lifespan_sum += max_lifespan
			max_lifespan_max = max(max_lifespan_max, max_lifespan)
			print("lifespans", lifespans)
			print("average lifespan", sum(lifespans.values()) / len(lifespans))
		avg_lifespan_mean = avg_lifespan_sum / batch_size
		max_lifespan_mean = max_lifespan_sum / batch_size
		log = {
			'avg_lifespan_mean': avg_lifespan_mean,
			'max_lifespan_mean': max_lifespan_mean,
			'max_lifespan_max': max_lifespan_max,
		}
		print(log)
		logger = Logger(policyName)
		logger.log(log)
