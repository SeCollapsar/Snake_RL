import os
import numpy as np

from env.snake_env import SnakeEnv
from rl.policy_network import PolicyNetwork
from rl.reinforce import ReinforceAgent


env = SnakeEnv()

policy = PolicyNetwork()

if os.path.exists("models/policy.npy"):
    policy.load()

agent = ReinforceAgent(policy)

episodes = 5000


for ep in range(episodes):

    state = env.reset()

    states = []
    actions = []
    rewards = []
    probs = []
    hiddens = []

    while True:

        action, p, h = agent.sample_action(state)

        next_state, reward, done = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        probs.append(p)
        hiddens.append(h)

        state = next_state

        if done:
            break

    agent.update(states, actions, probs, hiddens, rewards)

    if ep % 100 == 0:
        print("Episode", ep, "Reward", sum(rewards))

        policy.save()