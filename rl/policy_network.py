import numpy as np
import os
from datetime import datetime

from config import Config


class PolicyNetwork:

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

    # =========================
    # 主模型保存（覆盖）
    # =========================
    def save(self):

        grid = Config.GRID_SIZE

        os.makedirs(Config.MODEL_DIR, exist_ok=True)

        path = os.path.join(
            Config.MODEL_DIR,
            f"policy_{grid}.npy"
        )

        np.save(path, {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2
        })

    # =========================
    # 加载主模型
    # =========================
    def load(self):

        grid = Config.GRID_SIZE

        path = os.path.join(
            Config.MODEL_DIR,
            f"policy_{grid}.npy"
        )

        if not os.path.exists(path):
            print("[INFO] No existing model found, training from scratch.")
            return

        data = np.load(path, allow_pickle=True).item()

        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]

        print(f"[INFO] Loaded model: {path}")

    # =========================
    # 备份最优模型
    # =========================
    def save_backup(self, score):

        grid = Config.GRID_SIZE

        backup_dir = os.path.join(
            Config.MODEL_DIR,
            f"policy_{grid}_backup"
        )

        os.makedirs(backup_dir, exist_ok=True)

        # ---------- 读取已有模型 ----------
        existing = []

        for f in os.listdir(backup_dir):
            if f.endswith(".npy"):
                try:
                    # 文件名: policy_10_8.50_20260325.npy
                    parts = f.replace(".npy", "").split("_")
                    reward = float(parts[2])
                    existing.append((reward, f))
                except:
                    continue

        # ---------- 判断是否需要保存 ----------
        if len(existing) >= Config.MAX_BACKUP_MODELS:

            # 找最差的
            existing.sort(key=lambda x: x[0])  # 按reward排序

            worst_reward, worst_file = existing[0]

            if score <= worst_reward:
                # ❌ 不够优秀，直接丢弃
                return

            # 删除最差模型
            os.remove(os.path.join(backup_dir, worst_file))
            print(f"[REMOVE WORST] {worst_file}")

        # ---------- 保存新模型 ----------
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"policy_{grid}_{score:.2f}_{time_str}.npy"

        path = os.path.join(backup_dir, filename)

        np.save(path, {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2
        })

        print(f"[TOP-K SAVE] {filename}")