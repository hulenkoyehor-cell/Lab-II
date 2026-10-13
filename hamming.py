import numpy as np

class HammingNetwork:
    def __init__(self, patterns: np.ndarray):
        # записуємо еталон
        self.m, self.n = patterns.shape  
        self.W1 = patterns / 2.0
        self.b1 = self.n / 2.0

    def predict(self, x_input: np.ndarray, max_iter: int = 50):
        # схожість з кожним елементом
        y1 = np.dot(self.W1, x_input) + self.b1
        
        # відбирає відповіді
        y2 = np.copy(y1)
        eps = 0.4 / self.m

        for _ in range(max_iter):
            y2_prev = np.copy(y2)
            sum_all = np.sum(y2_prev)
            y2 = np.maximum(0, y2_prev - eps * (sum_all - y2_prev))
            
            # зупиняється коли залишилась одна правельна відповідь
            if np.count_nonzero(y2) <= 1 or np.array_equal(y2, y2_prev):
                break

        winning_class = int(np.argmax(y2))
        return winning_class, y1