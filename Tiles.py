from enum import Enum

class Tiles(Enum):
	NULL = 0
	GRASS  = 1
	FOREST_DEPLETED = 2
	FOREST = 3
	WATER  = 4

	def isOccupiable(tileValue: int):
		return Tiles(tileValue) not in [Tiles.NULL, Tiles.WATER]