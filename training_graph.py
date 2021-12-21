import matplotlib.pyplot as plt
from logger import Logger
import os

trainings = [
	("bc", "BC"),
	("pg", "PG"),
	("ppo", "PPO"),
	("ppo_lstm", "PPO w/ LSTM"),
	("ppo_attention", "PPO w/ AttentionNet"),
]

print("Alg", "Epochs", "Time", "Avg. Lifespan", "Max Lifespan")
for training in trainings:
	log_name = training[0]
	log_title = training[1]

	logs_path = os.path.join('logs', log_name, log_name + '0.log')
	training_logs = Logger.read(logs_path)
	from datetime import datetime
	training_time = datetime.fromtimestamp(training_logs[-1]['timestamp']) - datetime.fromtimestamp(training_logs[0]['timestamp'])
	print(log_title, len(training_logs), training_time, training_logs[-1]['custom_metrics']['avg_lifespan_mean'], training_logs[-1]['custom_metrics']['max_lifespan_mean'])
	

	avg_lifespans = [training_log['custom_metrics']['avg_lifespan_mean'] for training_log in training_logs]
	max_lifespans = [training_log['custom_metrics']['max_lifespan_mean'] for training_log in training_logs]
	epochs = range(1,len(training_logs)+1)
	fig = plt.figure(dpi=1200)
	plt.plot(epochs, avg_lifespans, 'g', label='Avg. Lifespans')
	plt.plot(epochs, max_lifespans, 'b', label='Max Lifespans')
	plt.title(log_title + " Lifespans over Training")
	plt.xlabel('Epochs')
	plt.ylabel('Agent Iters Alive')
	plt.legend()
	# plt.show()
	plt.savefig(log_name + "_fig.png")
	plt.close(fig)
