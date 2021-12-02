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
