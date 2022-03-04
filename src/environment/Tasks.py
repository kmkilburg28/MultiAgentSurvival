from enum import Enum

# movement, direction, action (None, drink water (direction), consume food from ground, consume food from hands, drop food, attack (direction))
class Tasks(Enum):
	NONE = 0
	DRINK = 1
	CONSUME_GROUND = 2
	CONSUME_CARRY = 3
	PICK_UP  = 4
	DROP = 5
	ATTACK = 6 # drops food if carrying
