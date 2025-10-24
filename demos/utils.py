import exceptiongroup
import numpy as np


def softmax(z: np.array):
    total_sum = sum(np.exp(z))
    return np.exp(z) / total_sum

def softmax_gpt(z: np.ndarray) -> np.ndarray:
    # z peut être (C,) ou (N, C)
    z = np.asarray(z)
    if z.ndim == 1:
        z_shift = z - np.max(z)
        e = np.exp(z_shift)
        return e / np.sum(e)
    elif z.ndim == 2:
        z_shift = z - np.max(z, axis=1, keepdims=True)  # par ligne
        e = np.exp(z_shift)
        return e / np.sum(e, axis=1, keepdims=True)
    else:
        raise ValueError("softmax attend un vecteur (C,) ou une matrice (N, C)")


def cross_entropy_batch(probs: np.ndarray, y: np.ndarray) -> float:
    # probs: (N, C), y: (N,)
    eps = 1e-12
    p = probs[np.arange(y.size), y]
    return -np.mean(np.log(p + eps))



class Trainer:
    def __init__(self):
        # __init__
        self.W = np.zeros((1, 2), dtype=float)  # (D=1, C=2)
        self.b = np.zeros((2,), dtype=float)

        self.learning_rate = 0.05

    def train(self, x, y):
        # x: (1,)  /  W: (1,2)
        logits = x @ self.W + self.b  # (2,)   ← pas de "*"
        probs = softmax_gpt(logits)  # (2,)
        loss = -np.log(probs[y] + 1e-12)

        one_hot = np.array([0., 0.]);
        one_hot[y] = 1.
        delta = probs - one_hot  # (2,)

        dW = np.outer(x, delta)  # (1,2)
        db = delta  # (2,)

        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

    def train_epoch(self, X, y):
        score = X @ self.W + self.b
        probs = softmax_gpt(score)

        loss = cross_entropy_batch(probs, y)

        # one-hot (N,2)
        Y = np.zeros((y.size, 2), dtype=float)
        Y[np.arange(y.size), y] = 1.0

        delta = probs - Y  # (N,2)

        N = X.shape[0]
        dW = (X.T @ delta) / N  # (D,2)
        db = delta.mean(axis=0)  # (2,)

        # update
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

        # (optionnel) accuracy pour suivi
        preds = np.argmax(probs, axis=1)
        acc = (preds == y).mean()

        print(f"Loss: {loss:.6f} | Acc: {acc:.3f}")

    def test(self, x):
        logits = x @ self.W + self.b  # (2,)
        probs = softmax_gpt(logits)
        return int(np.argmax(probs))


X_train = np.array([[0.00],
                    [0.10],
                    [0.30],
                    [0.49],
                    [0.51],
                    [0.70],
                    [0.85],
                    [1.00]], dtype=float).reshape(-1, 1)        # shape (8,1)
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])       # shape (8,)

X_val = np.array([[0.25],
                  [0.55]], dtype=float)
y_val = np.array([0, 1])

trainer = Trainer()
#for (x, y) in zip(X_train, y_train):
#    trainer.train(x, y)
for i in range(10000):
    trainer.train_epoch(X_train, y_train)

print("-------")

for (x, y) in zip(X_val, y_val):
    print(trainer.test(x), y)
