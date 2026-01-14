import numpy as np
from tqdm import tqdm
import pickle

class Toy:
    def _get_prob(self, group_num, j):
        if group_num == 1 and j < self.block_size:
            return 0.1
        elif group_num == 2 and j >= self.block_size:
            return 0.1
        elif group_num == 3 and j < int(self.block_size/4):
            return 0.4
        elif group_num == 4 and (j >= int(self.block_size/4)) and (j < int(self.block_size/2)):
            return 0.4
        elif group_num == 5 and (j >= int(self.block_size/2)) and (j < int(3*self.block_size/4)):
            return 0.4
        elif group_num == 6 and (j >= int(3*self.block_size/4)) and (j < self.block_size):
            return 0.4
        elif group_num == 7 and j >= self.block_size:
            return 0.1
        return 0

    def _get_group_num(self, i):
        if i < int(self.n / 4):
            return 1
        elif i < int(self.n / 2):
            return 2
        elif i < int(9*self.n / 16):
            return 3
        elif i < int(10*self.n / 16):
            return 4
        elif i < int(11*self.n / 16):
            return 5
        elif i < int(12*self.n / 16):
            return 6
        return 7

    def __init__(self, num_users=1000, item_block_size=100, seed=0):
        self.block_size = item_block_size
        self.n = num_users

        rng = np.random.default_rng(seed=seed)

        X = np.zeros((self.n, 2*self.block_size))
        for i in range(self.n):
            group_num = self._get_group_num(i)
            for j in range(2 * self.block_size):
                p = self._get_prob(group_num, j)
                X[i, j] = rng.binomial(n=1, p=p)
        self.X = X

    def get_X(self):
        return self.X

    def get_user_labels(self):
        return {
            "Group 1": np.array(range(int(self.n / 2))),
            "Group 2": np.array(range(int(self.n / 2), self.n))
        }
        

if __name__ == "__main__":
    toy_obj = Toy()