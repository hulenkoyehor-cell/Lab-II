import numpy as np

class RosenblattPerceptron:
    def __init__(self, input_dim: int, learning_rate: float = 0.1):
        self.lr = learning_rate
        # початкові ваги та поріг
        self.w = np.random.uniform(0.01, 0.99, size=input_dim)
        self.b = np.random.uniform(0.01, 0.99)
        self.correct_steps = 0  

    def predict(self, x: np.ndarray) -> int:
        # множимо суму ваги та віднімаємо поріг
        net = np.dot(self.w, x) - self.b
        return 1 if net > 0 else -1

    def train_step(self, x: np.ndarray, d: int):
        # поточеа відповідь 
        y = self.predict(x)
        error = d - y  

        if error != 0:
            # якщо помилка змінюємо ваги та поріг
            self.w += self.lr * error * x
            self.b -= self.lr * error
            self.correct_steps = 0
            return y, f"Помилка! Ваги скориговано. Новий b = {self.b:.2f}"
        else:
            # якщо вгадали нічого не міняємо
            self.correct_steps += 1
            return y, f"Вірно! Ваги не змінювалися. Послідовних успіхів: {self.correct_steps}"