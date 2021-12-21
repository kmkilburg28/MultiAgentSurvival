
from gym import spaces
from gym.utils import seeding
import pygame
import functools
import numpy as np
from random import randrange

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers

from Config import Config
from Direction import Direction
from Tasks import Tasks
from Tiles import Tiles
from Agent import Agent
from Map import Map


NONE = 0

def env(config: Config):
	env = raw_env(config)
	env = wrappers.CaptureStdoutWrapper(env)
	# env = wrappers.AssertOutOfBoundsWrapper(env)
	env = wrappers.OrderEnforcingWrapper(env)
	return env
	
class raw_env(AECEnv):
	metadata = {'render.modes': ['human'], "name": "survival_world_v0"}
	
	def __init__(self, config: Config):
		self.config = config
		self.possible_agents = ["player_" + str(i) for i in range(config.NUM_AGENTS)]
		self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
		self.closed = False
		self.has_reset = False
		self.seed()

		observation_width = 2*config.AGENT_VIEW_RADIUS+1
		self._observation_spaces = {
			agent: spaces.Dict({
				'grid'   : spaces.Box(low=0, high=255, shape=(observation_width, observation_width), dtype=np.int32),
				'dropped': spaces.Box(low=0, high=255, shape=(observation_width, observation_width), dtype=np.int32),
				'agents' : spaces.Box(low=0, high=255, shape=(observation_width, observation_width), dtype=np.int32),
				'status' : spaces.Box(low=0, high=self.config.MAX_HEALTH, shape=(3,), dtype=np.int32),
			}) for agent in self.possible_agents
		}
		dir_options = max([direction.value for direction in list(Direction)])+1
		task_options = max([task.value for task in list(Tasks)])+1
		self._action_spaces = {agent: spaces.MultiDiscrete([2, dir_options, task_options]) for agent in self.possible_agents} # movement, action (None, drink water (direction), consume food from ground, consume food from hands, drop food, attack (direction))
		# self._action_spaces = {agent: spaces.Tuple([spaces.Discrete(2), spaces.Discrete(dir_options), spaces.Discrete(task_options)]) for agent in self.possible_agents} # movement, action (None, drink water (direction), consume food from ground, consume food from hands, drop food, attack (direction))

	@functools.lru_cache(maxsize=None)
	def observation_space(self, agent):
		return self._observation_spaces[agent]

	@functools.lru_cache(maxsize=None)
	def action_space(self, agent):
		return self._action_spaces[agent]

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)

	def render(self, mode="human"):
		pygame.display.init()
		self.screen = pygame.display.set_mode(
			(self.config.TILE_SCALE * self.config.SIZE, self.config.TILE_SCALE * self.config.SIZE))
		br = self.config.AGENT_VIEW_RADIUS # border radius

		# Draw base tiles
		for row in range(self.config.SIZE):
			for col in range(self.config.SIZE):
				pos = pygame.Rect(
					self.config.TILE_SCALE * col, self.config.TILE_SCALE * row, self.config.TILE_SCALE, self.config.TILE_SCALE)
				color = (0,0,0)
				tile = self.map.active_grid[row+br,col+br]
				if tile == Tiles.GRASS.value:
					color = (0,255,0)
				elif tile == Tiles.FOREST.value:
					color = (0,128,0)
				elif tile == Tiles.FOREST_DEPLETED.value:
					color = (128,128,0)
				elif tile == Tiles.WATER.value:
					color = (0,0,255)
				elif tile == Tiles.NULL.value:
					color = (64,64,64)
				pygame.draw.rect(self.screen, color, pos)

		# Draw border lines between tiles
		color = (0,0,0)
		for row in range(self.config.SIZE):
			pos = pygame.Rect(
				0, self.config.TILE_SCALE * row-1, self.config.TILE_SCALE*self.config.SIZE, 2)
			pygame.draw.rect(self.screen, color, pos)
		for col in range(self.config.SIZE):
			pos = pygame.Rect(
				self.config.TILE_SCALE * col-1, 0, 2, self.config.TILE_SCALE*self.config.SIZE)
			pygame.draw.rect(self.screen, color, pos)

		# Draw dropped food
		pygame.font.init()
		def RenderDroppedFood(row: int, col: int, amount: int):
			default_font_path = pygame.font.get_default_font()
			maxFontSize = int(self.config.TILE_SCALE * 0.2)
			font = pygame.font.Font(default_font_path, maxFontSize)
			text = font.render(str(amount), False, (0,0,0))

			cornerRow = (row+0.05)*self.config.TILE_SCALE
			cornerCol = (col+0.05)*self.config.TILE_SCALE
			self.screen.blit(text, (cornerCol, cornerRow))

		for row in range(self.config.SIZE):
			for col in range(self.config.SIZE):
				if self.map.dropped_grid[row+br,col+br] > 0:
					RenderDroppedFood(row, col, self.map.dropped_grid[row+br,col+br])

		# Draw agents
		def RenderAgent(agent_instance: Agent):
			points = []
			height = 0.8
			radius = height / 2

			if agent_instance.direction == Direction.SOUTH.value or agent_instance.direction == Direction.NORTH.value:
				radius = radius if agent_instance.direction == Direction.SOUTH.value else -radius
				points = [
					(0.5-radius,0.5-radius),
					(0.5+radius,0.5-radius),
					(0.5, 0.5+radius)
				]
			elif agent_instance.direction == Direction.EAST.value or agent_instance.direction == Direction.WEST.value:
				radius = radius if agent_instance.direction == Direction.EAST.value else -radius
				points = [
					(0.5-radius,0.5-radius),
					(0.5-radius,0.5+radius),
					(0.5+radius, 0.5)
				]
			points = [
				(self.config.TILE_SCALE*(point[0] + agent_instance.loc[1]-br), self.config.TILE_SCALE*(point[1] + agent_instance.loc[0]-br)) 
				for point in points
			]
			color = (127,0,0)
			pygame.draw.polygon(self.screen, color, points)

			# Draw food if carrying
			if agent_instance.carrying:
				color = (0,128,0)
				center = (self.config.TILE_SCALE*(agent_instance.loc[1]+0.5-br), self.config.TILE_SCALE*(agent_instance.loc[0]+0.5-br))
				radius = (height / 6)*self.config.TILE_SCALE
				pygame.draw.circle(self.screen, color, center, radius)

		for agent in self.agents:
			agent_instance = self.agent_instances[agent]
			RenderAgent(agent_instance)
		

	def observe(self, agent: str):
		agent_instance = self.agent_instances[agent]
		radius = self.config.AGENT_VIEW_RADIUS
		row, col = agent_instance.loc
		minR = row - radius
		maxR = row + radius+1
		minC = col - radius
		maxC = col + radius+1
		agents_grid = self.map.agents_grid[minR:maxR, minC:maxC]
		status = np.array([
			agent_instance.health,
			agent_instance.food,
			agent_instance.water,
		], dtype=np.uint8)
		return {
			'grid'   : self.map.active_grid[minR:maxR, minC:maxC],
			'dropped': self.map.dropped_grid[minR:maxR, minC:maxC],
			'agents' : agents_grid,
			'status' : status,
		}

	def close(self):
		if not self.closed:
			self.closed = True

	def reset(self):
		self.has_reset = True
		self.map = Map.generate(self.config)
		self.agents = self.possible_agents[:]
		self.agent_location_mappings = {}
		self.agent_instances = {}
		if self.config.NUM_AGENTS > self.config.SIZE**2:
			raise Exception("Too many agents. Number of agents exeeds amount of occupiable tiles.")
		for agent in self.agents:
			not_placed = True
			while not_placed:
				row = randrange(0, self.map.ROWS)
				col = randrange(0, self.map.COLS)
				if self.map.isOccupiable(row, col):
					loc = (row, col)
					agent_instance = Agent(agent, loc, self.config)
					self.agent_instances[agent] = agent_instance
					self.agent_location_mappings[loc] = agent
					self.map.agents_grid[row,col] = 1
					not_placed = False
		self.rewards = {agent: 0 for agent in self.agents}
		self._cumulative_rewards = {agent: 0 for agent in self.agents}
		self.dones = {agent: False for agent in self.agents}
		self.infos = {agent: {} for agent in self.agents}
		self.agents_alive = len(self.agents)
		self.num_moves = 0

		self._agent_selector = agent_selector(self.agents)
		self.agent_selection = self._agent_selector.next()
		observations = {agent: self.observe(agent) for agent in self.agents}
		return observations

	def killAgent(self, agent: str):
		agent_instance = self.agent_instances[agent]
		agent_instance.health = 0
		agent_instance.food = 0
		agent_instance.water = 0
		self.agent_location_mappings.pop(agent_instance.loc)
		self.map.agents_grid[agent_instance.loc] = 0
		self.agents_alive -= 1
		agent_instance.done = True
		self.rewards[agent] = -1
		self.dones[agent] = True

	def step(self, action: np.ndarray):
		agent = self.agent_selection
		agent_instance = self.agent_instances[agent]
		# for forest_loc in agent_instance.depleted_forests:
		# 	agent_instance.depleted_forests[forest_loc] -= 1
		# 	if agent_instance.depleted_forests[forest_loc] <= 0:
		# 		self.map.active_grid[forest_loc] = Tiles.FOREST.value
		if self.dones[agent]:
			self._was_done_step(action)
			if self.agents_alive > 0 and self.agent_selection not in self.agents:
				self.agent_selection = self._agent_selector.next()
			return

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
				if other_agent_instance.done:
					self.killAgent(other_agent)
		elif task == Tasks.DRINK.value:
			dstRow, dstCol = Direction.getNextTile(row, col, direction)
			if self.map.active_grid[dstRow,dstCol] == Tiles.WATER.value:
				agent_instance.drink(self.config.DRINK_SIZE)
				# if agent_instance.water < self.config.MAX_WATER - 2:
				# 	self.rewards[agent] += 0.001
		elif task == Tasks.CONSUME_GROUND.value:
			if self.map.active_grid[row,col] == Tiles.FOREST.value:
				agent_instance.eat(self.config.FOOD_SIZE)
				self.map.active_grid[row,col] = Tiles.FOREST_DEPLETED.value
				# if agent_instance.food < self.config.MAX_FOOD - 2:
				# 	self.rewards[agent] += 0.001
				# agent_instance.depleted_forest[(row,col)] = self.config.FOREST_RENEW_TURNS
		elif task == Tasks.CONSUME_CARRY.value:
			if agent_instance.carrying:
				agent_instance.eat(self.config.FOOD_SIZE)
				self.map.agents_grid[row,col] = 1
				agent_instance.carrying = False
				# if agent_instance.food < self.config.MAX_FOOD - 2:
				# 	self.rewards[agent] += 0.001
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
		self.agent_selection = self._agent_selector.next()
		
		# Adds .rewards to ._cumulative_rewards
		self._accumulate_rewards()
		self._clear_rewards()
