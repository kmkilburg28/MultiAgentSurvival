import sys
import os
sys.path.append(os.getcwd())

from src.environment.Direction import Direction
from src.environment.Tasks import Tasks
from src.models.utils.run_policy import run_policy

# Idle policy
def policy(observation, agent):
	return [0,Direction.SOUTH.value,Tasks.NONE.value]

if __name__ == "__main__":
	run_policy(policy)
