import numpy as np
from queue import PriorityQueue

import sys
import os
sys.path.append(os.getcwd())

from src.environment.Agent import Agent
from src.environment.Tiles import Tiles
from src.environment.Config import Config
from src.environment.virtual.SearchEnv import SearchEnv, Node
from src.models.utils.run_policy import run_policy


def searchHeuristic(searchEnv: SearchEnv):
	agent = searchEnv.agent_selection
	agent_instance: Agent = searchEnv.agent_instances[agent]

	agentLoc = agent_instance.loc
	# Water
	locs = np.where(searchEnv.map.active_grid == Tiles.WATER.value)
	waterDistances = []
	for i in range(len(locs[0])):
		waterDistances += [abs(agentLoc[0]-locs[0][i]) + abs(agentLoc[1]-locs[1][i])]
	minWaterDist = 4*searchEnv.config.AGENT_VIEW_RADIUS if len(waterDistances) <= 0 else min(waterDistances) - 1

	# Food
	locs = np.where(searchEnv.map.active_grid == Tiles.FOREST.value) + \
			np.where(searchEnv.map.dropped_grid > 0)
	foodDistances = []
	for i in range(len(locs[0])):
		foodDistances += [abs(agentLoc[0]-locs[0][i]) + abs(agentLoc[1]-locs[1][i])]
	minFoodDist = 4*searchEnv.config.AGENT_VIEW_RADIUS if len(foodDistances) <= 0 else min(foodDistances)

	return (
			4**(searchEnv.config.MAX_HEALTH - agent_instance.health) + \
			2**(searchEnv.config.MAX_FOOD - agent_instance.food)*minFoodDist + \
			2**(searchEnv.config.MAX_WATER - agent_instance.water)*minWaterDist
	)

def evaluationHeuristic(searchEnv: SearchEnv, depth: int):
	return -searchHeuristic(searchEnv)
def evaluationHeuristic2(searchEnv: SearchEnv, depth: int):
	agent_instance = searchEnv.agent_instances[searchEnv.agent_selection]
	return 10*(depth + agent_instance.health) + agent_instance.food + agent_instance.water


def SearchPolicy(obs, searchH, evaluationH, max_depth):
	initialSearchEnv = SearchEnv(Config(), obs)
	agent = initialSearchEnv.agent_selection
	INITIAL_STATE = initialSearchEnv.getState(agent)
	ROOT = Node(None, None, INITIAL_STATE, initialSearchEnv)
	num_nodes = 1

	frontier = PriorityQueue()
	score = searchH(initialSearchEnv)
	frontier.put((score, ROOT))
	reached = {INITIAL_STATE}

	num_nodes = 1
	bestFinalNode = None
	MIN_SCORE = -1000000000
	best_score = MIN_SCORE
	while not frontier.empty():
		_parent_score, parent = frontier.get()
		done = parent.searchEnv.agent_instances[agent].done
		if parent.depth >= max_depth or done:
			eval_score = ((MIN_SCORE + 1 + parent.depth) if done else evaluationH(parent.searchEnv, parent.depth))
			if best_score < eval_score:
				bestFinalNode = parent
				best_score = eval_score
		else:
			legalActions = parent.searchEnv.getLegalActions(agent)
			for action in legalActions:
				child_searchEnv = parent.searchEnv.clone()
				num_nodes += 1
				if child_searchEnv.step(action):
					child_state = child_searchEnv.getState(agent)
					if child_state not in reached:
						reached.add(child_state)
						score = searchH(child_searchEnv)
						node = Node(parent, action, child_state, child_searchEnv)
						frontier.put((score, node))
	
	return bestFinalNode.getOriginAction()

def policy(observation, agent):
    return SearchPolicy(observation, searchHeuristic, evaluationHeuristic2, 4)

if __name__ == "__main__":
	run_policy(policy)
