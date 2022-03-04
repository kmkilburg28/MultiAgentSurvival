
import numpy as np

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

from src.environment.Config import Config
from src.environment.Direction import Direction
from src.environment.Tasks import Tasks
from src.environment.Tiles import Tiles
from src.environment.Agent import Agent
from src.environment.Map import Map

	
class SearchEnv(AECEnv):
	
	def __init__(self, config: Config, obs: dict, searchEnv=None):
		if searchEnv:
			self.config = searchEnv.config
			self.map = searchEnv.map.clone()
			self.agents = searchEnv.agents.copy()
			self.agent_instances = {agent: searchEnv.agent_instances[agent].clone() for agent in searchEnv.agents}
			self.agent_location_mappings = searchEnv.agent_location_mappings.copy()
			self._agent_selector = agent_selector(self.agents)
			self._agent_selector._current_agent = searchEnv._agent_selector._current_agent
			self._agent_selector.selected_agent = searchEnv._agent_selector.selected_agent
			self.agent_selection = searchEnv.agent_selection
			self.agents_alive = searchEnv.agents_alive
			return
		self.config = config
		self.map = Map.createFromObservation(obs)
		self.constructAgents()

		agent_instance = self.agent_instances[self.agent_selection]
		agent_instance.health = obs["status"][0]
		agent_instance.food = obs["status"][1]
		agent_instance.water = obs["status"][2]

	def constructAgents(self):
		agent_locs = np.nonzero(self.map.agents_grid)
		locs = []
		if len(agent_locs) > 0:
			for i in range(len(agent_locs[0])):
				locs += [(agent_locs[0][i], agent_locs[1][i])]
		agent_locs = locs
		
		self.agents = ["player_" + str(i) for i in range(len(agent_locs))]
		self.agent_instances = {}
		self.agent_location_mappings = {}
		for i in range(len(self.agents)):
			agent = self.agents[i]
			loc = agent_locs[i]
			agent_instance = Agent(agent, loc, self.config)
			agent_instance.carrying = self.map.agents_grid[loc] > 1
			self.agent_instances[agent] = agent_instance
			self.agent_location_mappings[loc] = agent

		self._agent_selector = agent_selector(self.agents)
		self.agent_selection = self._agent_selector.next()
		while self.agent_selection != agent:
			self.agent_selection = self._agent_selector.next()
		self.agents_alive = len(self.agents)

	def observe(self, agent: str):
		agent_instance = self.agent_instances[agent]
		status = np.array([
			agent_instance.health,
			agent_instance.food,
			agent_instance.water,
		], dtype=np.uint8)
		return {
			'grid'   : self.map.active_grid,
			'dropped': self.map.dropped_grid,
			'agents' : self.map.agents_grid,
			'status' : status,
		}

	def killAgent(self, agent: str):
		agent_instance = self.agent_instances[agent]
		agent_instance.health = 0
		agent_instance.food = 0
		agent_instance.water = 0
		self.agent_location_mappings.pop(agent_instance.loc)
		self.map.agents_grid[agent_instance.loc] = 0
		self.agents_alive -= 1
		agent_instance.done = True


	def step(self, action: np.ndarray):
		agent = self.agent_selection
		agent_instance = self.agent_instances[agent]

		# for forest_loc in agent_instance.depleted_forests:
		# 	agent_instance.depleted_forests[forest_loc] -= 1
		# 	if agent_instance.depleted_forests[forest_loc] <= 0:
		# 		self.map.active_grid[forest_loc] = Tiles.FOREST.value
		if agent_instance.done:
			self.agents.pop(agent)
			if self.agents_alive > 0 and self.agent_selection not in self.agents:
				self.agent_selection = self._agent_selector.next()
			return False

		movement = action[0]
		direction = action[1]
		agent_instance.direction = direction
		task = action[2]
		row, col = agent_instance.loc
		if task == Tasks.ATTACK.value:	
			dstRow, dstCol = Direction.getNextTile(row, col, direction)
			if self.map.agents_grid[dstRow,dstCol] != 0:
				other_agent = self.agent_location_mappings[dstRow,dstCol]
				other_agent_instance = self.agent_instances[other_agent]
				agent_instance.attack(other_agent_instance)
				self.killAgent(other_agent)
		elif task == Tasks.DRINK.value:
			dstRow, dstCol = Direction.getNextTile(row, col, direction)
			if self.map.active_grid[dstRow,dstCol] == Tiles.WATER.value:
				agent_instance.drink(self.config.DRINK_SIZE)
		elif task == Tasks.CONSUME_GROUND.value:
			if self.map.active_grid[row,col] == Tiles.FOREST.value:
				agent_instance.eat(self.config.FOOD_SIZE)
				self.map.active_grid[row,col] = Tiles.FOREST_DEPLETED.value
				# agent_instance.depleted_forest[(row,col)] = self.config.FOREST_RENEW_TURNS
		elif task == Tasks.CONSUME_CARRY.value:
			if agent_instance.carrying:
				agent_instance.eat(self.config.FOOD_SIZE)
				self.map.agents_grid[row,col] -= 1
				agent_instance.carrying = False
		elif task == Tasks.PICK_UP.value:
			if not agent_instance.carrying:
				if self.map.dropped_grid[row,col] > 0:
					agent_instance.carrying = True
					self.map.dropped_grid[row,col] -= self.config.FOOD_SIZE
				elif self.map.active_grid[row,col] == Tiles.FOREST.value:
					agent_instance.carrying = True
					self.map.active_grid[row,col] = Tiles.FOREST_DEPLETED.value
					# agent_instance.depleted_forest[(row,col)] = self.config.FOREST_RENEW_TURNS
				if agent_instance.carrying:
					self.map.agents_grid[row,col] += 1
		if task == Tasks.DROP.value or task == Tasks.ATTACK.value:
			if agent_instance.carrying:
				agent_instance.carrying = False
				self.map.dropped_grid[row,col] += self.config.FOOD_SIZE
				self.map.agents_grid[row,col] = 1

		if movement:
			dstRow, dstCol = Direction.getNextTile(row, col, direction)
			if self.map.isOccupiable(dstRow, dstCol):
				self.agent_location_mappings.pop(agent_instance.loc)
				agent_instance.loc = (dstRow,dstCol)
				self.agent_location_mappings[agent_instance.loc] = agent
				self.map.agents_grid[dstRow,dstCol] = self.map.agents_grid[row,col]
				self.map.agents_grid[row,col] = 0

		if agent_instance.water <= 0:
			agent_instance.health -= 1
		else:
			agent_instance.water -= 1

		if agent_instance.food <= 0:
			agent_instance.health -= 1
		else:
			agent_instance.food -= 1

		if agent_instance.health <= 0:
			self.killAgent(agent)

		# selects the next agent.
		# self.agent_selection = self._agent_selector.next()
		return True

	def getLegalActions(self, agent: str):
		agent_instance = self.agent_instances[agent]
		if agent_instance.done:
			return [(0, 0, Tasks.NONE.value)]

		row, col = agent_instance.loc
		actions = [(0,0,Tasks.NONE.value)]
		groundConsumable = self.map.active_grid[row,col] == Tiles.FOREST.value or self.map.dropped_grid[row,col] > 0
		if agent_instance.carrying:
			actions += [(0,0,Tasks.DROP.value), (0,0,Tasks.CONSUME_CARRY.value)]
		elif groundConsumable:
			actions += [(0,0,Tasks.PICK_UP.value)]
		if groundConsumable:
			actions += [(0,0,Tasks.CONSUME_GROUND.value)] 
		for dir in Direction._member_map_.values():
			dir = dir.value
			nextRow, nextCol = Direction.getNextTile(row, col, dir)
			if nextRow < 0 or self.map.ROWS <= nextRow or nextCol < 0 or self.map.COLS <= nextCol or \
				self.map.active_grid[nextRow,nextCol] == Tiles.NULL.value:
				continue
			elif self.map.active_grid[nextRow,nextCol] == Tiles.WATER.value:
				actions += [(0,dir,Tasks.DRINK.value)]
			elif self.map.agents_grid[nextRow,nextCol] > 0:
				actions += [(move,dir,Tasks.ATTACK.value) for move in range(2)]
			else:
				actions += [(1,dir,Tasks.NONE.value)]
				if agent_instance.carrying:
					actions += [(1,dir,Tasks.DROP.value), (1,dir,Tasks.CONSUME_CARRY.value)]
				elif groundConsumable:
					actions += [(1,dir,Tasks.PICK_UP.value)]
				if groundConsumable:
					actions += [(1,dir,Tasks.CONSUME_GROUND.value)]
		return actions

	def clone(self):
		return SearchEnv(self.config, {}, self)


	def getState(self, agent: str):
		return (np.array2string(self.map.active_grid) + np.array2string(self.map.dropped_grid) + np.array2string(self.map.agents_grid, ) + self.agent_instances[agent].getStatusString()).replace('\n','')

class Node:
	def __init__(self, parent, from_action: str, state: str, searchEnv: SearchEnv):
		self.parent = parent
		self.from_action = from_action
		self.state = state
		self.searchEnv = searchEnv
		self.depth = 0 if parent is None else parent.depth + 1

	def __str__(self):
		if self.parent == None:
			return ""
		return "{0}{1}".format(str(self.parent), self.from_action)
	
	def __lt__(self, other):
		return self.depth < other.depth

	def getOriginAction(self):
		if self.parent.parent is None:
			return self.from_action
		return self.parent.getOriginAction()
