import Env
from Config import Config

from random_policy import policy as RandomPolicy
from SearchPolicy import policy as SearchPolicy
from logger import Logger

policies = {
    "RandomPolicy": RandomPolicy,
    "SearchPolicy": SearchPolicy,
}

batch_size = 64


for policyName in policies:
	policy = policies[policyName]
	config = Config()
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
 