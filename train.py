from env.snake_env import SnakeEnv
from rl.policy_network import PolicyNetwork
from rl.reinforce import ReinforceAgent
from config import Config

from utils.logger import Logger
from utils.topk_analyzer import analyze_topk


env = SnakeEnv()
policy = PolicyNetwork()
policy.load()

agent = ReinforceAgent(policy)

logger = Logger()

episodes = Config.EPISODES
best_reward = -1e9


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

    policy.save()

    if total_reward > best_reward:
        best_reward = total_reward
        policy.save_backup(total_reward)

    logger.log(total_reward)

    if ep % 100 == 0:
        print(f"Episode {ep}, Reward: {total_reward}")

        logger.save_curve()
        analyze_topk()