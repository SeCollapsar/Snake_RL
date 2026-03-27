import os
import matplotlib.pyplot as plt
from config import Config


class Logger:

    def __init__(self):

        self.rewards = []

        self.log_dir = os.path.join("logs")
        os.makedirs(self.log_dir, exist_ok=True)

    def log(self, reward):

        self.rewards.append(reward)

    def save_curve(self):

        plt.figure()

        plt.plot(self.rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Curve")

        path = os.path.join(self.log_dir, "reward_curve.png")
        plt.savefig(path)

        plt.close()