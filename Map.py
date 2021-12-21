import numpy as np
from random import randrange
from Config import Config
from Tiles import Tiles

class Map:
	def __init__(self, grid: np.ndarray):
		# Static
		self.grid = grid
		self.ROWS = len(grid)
		self.COLS = len(grid[0])

		# Dynamic
		self.active_grid = grid.copy()
		self.dropped_grid = np.zeros(grid.shape, np.uint8)
		self.agents_grid = np.zeros(grid.shape, np.uint8)

	def isOccupiable(self, row: int, col: int):
		return Tiles.isOccupiable(self.active_grid[row, col]) and self.agents_grid[row, col] == 0

	def generate(config: Config):

		def createPatchAt(grid: np.ndarray, tileType: int, row: int, col: int):
			ROWS = grid.shape[0]
			COLS = grid.shape[1]
			patchSize = randrange(2, 5+1)
			patches = set()
			checked = set()
			frontier = {(row,col)}
			patchTilesFound = 1
			while patchTilesFound < patchSize and len(frontier) > 0:
				tileLoc = frontier.pop()
				if grid[tileLoc[0],tileLoc[1]] == Tiles.GRASS.value:
					patches.add(tileLoc)
					patchTilesFound += 1
				checked.add(tileLoc)

				for rowOffset in range(-1, 2):
					newRow = tileLoc[0] + rowOffset
					if 0 <= newRow and newRow < ROWS:
						for colOffset in range(-1, 2):
							newCol = tileLoc[1] + colOffset
							if 0 <= newCol and newCol < COLS:
								newTileLoc = (newRow, newCol)
								if newTileLoc not in checked:
									frontier.add(newTileLoc)
			for patchLoc in patches:
				grid[patchLoc[0],patchLoc[1]] = tileType.value

		size = config.SIZE
		radius = config.AGENT_VIEW_RADIUS
		grid = np.ones((size+2*radius, size+2*radius), np.uint8) * Tiles.GRASS.value
		for patch in [(config.FOREST_PATCHES, Tiles.FOREST), (config.WATER_PATCHES, Tiles.WATER)]:
			i = 0
			while i < patch[0]:
				row = randrange(0, size) + radius
				col = randrange(0, size) + radius
				if grid[row,col] == Tiles.GRASS.value:
					createPatchAt(grid, patch[1], row, col)
					i += 1
		grid[:radius,:] = Tiles.NULL.value
		grid[size+radius:,:] = Tiles.NULL.value
		grid[:,:radius] = Tiles.NULL.value
		grid[:,size+radius:] = Tiles.NULL.value

		return Map(grid)

	def createFromObservation(obs):
		"""Used for simulating an observation without additional map info.

		Args:
			obs ({"grid", "dropped", "agents"}): An agent's observation.

		Returns:
			Map: A new map constructed purely from the provided observation.
		"""
		# radius = config.AGENT_VIEW_RADIUS
		# shape = (
		# 	obs["grid"].shape[0] + 2*radius,
		# 	obs["grid"].shape[1] + 2*radius
		# )
		# grid = np.ones(shape, np.uint8) * Tiles.NULL.value
		# dropped = np.zeros(shape, np.uint8)
		# agents = np.zeros(shape, np.uint8)
		# for row in range(obs["grid"].shape[0]):
		# 	cur_row = row + radius
		# 	for col in range(obs["grid"].shape[1]):
		# 		cur_col = col + radius
		# 		grid[cur_row,cur_col] = obs["grid"][row,col]
		# 		dropped[cur_row,cur_col] = obs["grid"][row,col]
		# 		agents[cur_row,cur_col] = obs["grid"][row,col]
		# map = Map(grid)
		# map.dropped_grid = dropped
		# map.agents_grid = agents
		map = Map(obs["grid"])
		map.dropped_grid = obs["dropped"].copy()
		map.agents_grid = obs["agents"].copy()
		return map

	def clone(self):
		map = Map(self.active_grid)
		map.dropped_grid = self.dropped_grid.copy()
		map.agents_grid = self.agents_grid.copy()
		return map
