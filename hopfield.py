import numpy as np

class HopfieldNetwork:
    def __init__(self, size: int):
        self.size = size
        self.W = np.zeros((size, size))

    def train(self, patterns: list):
        self.W = np.zeros((self.size, self.size))
        for p in patterns:
            p_arr = np.array(p)
            self.W += np.outer(p_arr, p_arr)
            
        self.W /= self.size
        np.fill_diagonal(self.W, 0)  

    def predict(self, x_input: np.ndarray, max_iter: int = 15, mode: str = 'async'):
        y = np.copy(x_input).astype(int)
        
        if mode == 'async':
            
            for iteration in range(max_iter):
                y_old = np.copy(y)
                indices = np.random.permutation(self.size) 
                
                for idx in indices:
                    net_input = np.dot(self.W[idx], y)
                    if net_input > 0:
                        y[idx] = 1
                    elif net_input < 0:
                        y[idx] = -1
                    
                    
                if np.array_equal(y, y_old):
                    return y, iteration + 1
            return y, max_iter

        else:
            
            for iteration in range(max_iter):
                net_input = np.dot(self.W, y)
                y_next = np.where(net_input > 0, 1, np.where(net_input < 0, -1, y))
                
                if np.array_equal(y_next, y):
                    return y_next, iteration + 1
                y = y_next
            return y, max_iter

