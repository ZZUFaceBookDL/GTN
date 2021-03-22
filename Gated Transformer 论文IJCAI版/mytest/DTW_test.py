import numpy as np

from dtw import dtw

# x = np.array([2, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
# y = np.array([1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
x = np.array([8, 9, 1, 9, 6, 1, 3, 5]).reshape(-1, 1)
y = np.array([2, 5, 4, 6, 7, 8, 3, 7, 7, 2]).reshape(-1, 1)


euclidean_norm = lambda x, y: np.abs(x - y)

d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)

print('d ', d)
print('cost_matrix \r\n', cost_matrix)
print('acc_cost_matrix \r\n', acc_cost_matrix)
print('path ', path)