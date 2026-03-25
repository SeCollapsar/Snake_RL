import os

from env.snake_env import SnakeEnv
from rl.policy_network import PolicyNetwork
from rl.reinforce import ReinforceAgent
from config import Config


env = SnakeEnv()
policy = PolicyNetwork()
policy.load()

agent = ReinforceAgent(policy)

episodes = Config.EPISODES

best_reward = -1e9  # 记录历史最优


for ep in range(episodes):

    state = env.reset()

    states, actions, rewards, probs, hiddens = [], [], [], [], []

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

    total_reward = sum(rewards)

    agent.update(states, actions, probs, hiddens, rewards)

    # ---------- 保存主模型 ----------
    policy.save()

    # ---------- 保存最优模型 ----------
    if total_reward > best_reward:

        best_reward = total_reward

        print(f"[BEST] Episode {ep}, Reward: {total_reward}")

        policy.save_backup(best_reward)

    if ep % 100 == 0:
        print(f"Episode {ep}, Reward: {total_reward}")