import numpy as np


data = np.loadtxt("labs/02/gym_cartpole_data.txt")
observations, labels = data[:, :-1], data[:, -1].astype(np.int32)

print(np.max(observations, axis=0))
