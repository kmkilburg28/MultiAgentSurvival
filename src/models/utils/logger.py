import json
import os
import numpy as np

def np_encoder(object):
	if isinstance(object, np.generic):
		return object.item()

class Logger:
    
	def __init__(self, base):
		logs_dir = "logs"
		if not os.path.isdir(logs_dir):
			os.mkdir(logs_dir)
		base = os.path.basename(base)
		base_dir = os.path.join(logs_dir, base)
		if not os.path.isdir(base_dir):
			os.mkdir(base_dir)

		base_path = os.path.join(base_dir, base)
		ext = ".log"
		i = 0
		while os.path.exists(base_path + str(i) + ext):
			i += 1
		self.filepath = base_path + str(i) + ext

	def log(self, object):
		with open(self.filepath, "a+") as f:
			json_str = json.dumps(object, default=np_encoder)
			f.write(json_str + ",\n")

	def poll(self):
		return Logger.read(self.filepath)

	def read(filepath: str) -> list:
		with open(filepath, "r") as f:
			logs = f.read()
			object = json.loads("[" + logs[:-2] + "]")
			return object
