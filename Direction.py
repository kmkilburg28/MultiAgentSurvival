from enum import Enum

class Direction(Enum):
	NONE = 0
	SOUTH  = 1
	WEST = 2
	NORTH = 3
	EAST  = 4

	def getNextTile(row: int, col: int, direction: int):
		if Direction.SOUTH.value == direction:
			return row+1,col
		elif Direction.NORTH.value == direction:
			return row-1,col
		elif Direction.WEST.value == direction:
			return row,col-1
		elif Direction.EAST.value == direction:
			return row,col+1
		else:
			return row,col
