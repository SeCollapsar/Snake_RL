import numpy as np
from config import Config


class ReinforceAgent:

    def __init__(self, policy, lr=0.001, gamma=0.99):

        self.policy = policy
        self.lr = lr
        self.gamma = gamma

        # 防策略坍塌
        # self.epsilon = Config.EPSILON

    def sample_action(self, state):

        probs, hidden = self.policy.forward(state)

        # ---------- 防策略坍塌 ----------
        # probs = (1 - self.epsilon) * probs + self.epsilon / len(probs)
        # probs = probs / np.sum(probs)

        action = np.random.choice(len(probs), p=probs)

        return action, probs, hidden

    def compute_returns(self, rewards):

        returns = []

        G = 0

        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = np.array(returns)

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def update(self, states, actions, probs_list, hiddens, rewards):

        returns = self.compute_returns(rewards)

        for state, action, probs, hidden, G in zip(
            states, actions, probs_list, hiddens, returns
        ):

            dlog = -probs
            dlog[action] += 1

            dlog *= G

            dw2 = np.outer(hidden, dlog)
            db2 = dlog

            dh = np.dot(self.policy.w2, dlog)
            dh = (1 - hidden ** 2) * dh

            dw1 = np.outer(state, dh)
            db1 = dh

            self.policy.w2 += self.lr * dw2
            self.policy.b2 += self.lr * db2

            self.policy.w1 += self.lr * dw1
            self.policy.b1 += self.lr * db1