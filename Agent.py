from Config import Config
from Direction import Direction

MAX_HEALTH = 10
MAX_WATER  = 10
MAX_FOOD   = 10

ATTACK_DAMAGE = 5

class Agent:
    def __init__(self, name: str, spawnLoc: tuple, config: Config):
		# Static
        self.name = name
        self.spawnLoc = spawnLoc

        # Dynamic
        self.loc = self.spawnLoc
        self.health = MAX_HEALTH
        self.water  = MAX_WATER
        self.food   = MAX_FOOD
        self.direction = Direction.NORTH.value
        self.carrying = False
        self.done = False

        self.depleted_forests = {}

        self.kills = 0

    def attack(self, agent):
        agent.health -= ATTACK_DAMAGE
        if agent.health <= 0:
            agent.done = True
            self.kills += 1

    def drink(self, amount):
        self.water += amount
        if self.water > MAX_WATER:
            self.water = MAX_WATER
    def eat(self, amount):
        self.food += amount
        if self.food > MAX_FOOD:
            self.food = MAX_FOOD
