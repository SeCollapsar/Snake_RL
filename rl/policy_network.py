import numpy as np
import os


class PolicyNetwork:
    """
    两层神经网络
    输入: 300
    隐藏层: 128
    输出: 4
    """

    def __init__(self, input_dim=300, hidden=128, output=4):

        self.w1 = np.random.randn(input_dim, hidden) * 0.01
        self.b1 = np.zeros(hidden)

        self.w2 = np.random.randn(hidden, output) * 0.01
        self.b2 = np.zeros(output)

    def forward(self, x):

        h = np.dot(x, self.w1) + self.b1
        h = np.tanh(h)

        logits = np.dot(h, self.w2) + self.b2

        probs = self.softmax(logits)

        return probs, h

    def softmax(self, x):

        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    def save(self, path="models/policy.npy"):

        os.makedirs("models", exist_ok=True)

        np.save(path, {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2
        })

    def load(self, path="models/policy.npy"):

        data = np.load(path, allow_pickle=True).item()

        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]