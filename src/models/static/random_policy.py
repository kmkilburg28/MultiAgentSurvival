from random import randint

import sys
import os
sys.path.append(os.getcwd())

from src.environment.Direction import Direction
from src.environment.Tasks import Tasks
from src.models.utils.run_policy import run_policy

# Random policy
max_dir = max([direction.value for direction in list(Direction)])
max_task = max([task.value for task in list(Tasks)])
def policy(observation, agent):
	return [randint(0,1),randint(0,max_dir),randint(0,max_task)]

if __name__ == "__main__":
	run_policy(policy)
