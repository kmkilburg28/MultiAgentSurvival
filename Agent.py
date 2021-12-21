from Config import Config
from Direction import Direction

class Agent:

	def __init__(self, name: str, spawnLoc: tuple, config: Config):
		# Static
		self.name = name
		self.spawnLoc = spawnLoc
		self.config = config

		# Dynamic
		self.loc = self.spawnLoc
		self.health = self.config.MAX_HEALTH
		self.water  = self.config.MAX_WATER
		self.food   = self.config.MAX_FOOD
		self.direction = Direction.NORTH.value
		self.carrying = False
		self.done = False

		self.depleted_forests = {}

		self.kills = 0

	def attack(self, agent):
		agent.health -= self.config.ATTACK_DAMAGE
		if agent.health <= 0:
			agent.done = True
			self.kills += 1

	def drink(self, amount):
		self.water += amount
		if self.water > self.config.MAX_WATER:
			self.water = self.config.MAX_WATER
	def eat(self, amount):
		self.food += amount
		if self.food > self.config.MAX_FOOD:
			self.food = self.config.MAX_FOOD

	def clone(self):
		agent = Agent(self.name, self.spawnLoc, self.config)
		agent.loc = self.loc
		agent.health = self.health
		agent.food = self.food
		agent.water = self.water
		agent.direction = self.direction
		agent.carrying = self.carrying
		agent.done = self.done
		agent.kills = self.kills
		return agent

	def getStatusString(self):
		return "{0},{1},{2}".format(self.health, self.food, self.water)