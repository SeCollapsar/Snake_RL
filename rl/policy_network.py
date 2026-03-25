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

        dir_path = os.path.join(Config.MODEL_DIR)
        os.makedirs(dir_path, exist_ok=True)

        path = os.path.join(dir_path, f"policy_{grid}.npy")

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
            f"policy_{grid}",
            "policy.npy"
        )

        if not os.path.exists(path):
            return

        data = np.load(path, allow_pickle=True).item()

        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]

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

        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"policy_{grid}_{score:.2f}_{time_str}.npy"

        path = os.path.join(backup_dir, filename)

        np.save(path, {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2
        })

        # ---------- 清理旧模型 ----------
        self._cleanup_backup(backup_dir)

    def _cleanup_backup(self, backup_dir):

        files = [
            f for f in os.listdir(backup_dir)
            if f.endswith(".npy")
        ]

        if len(files) <= Config.MAX_BACKUP_MODELS:
            return

        # 按时间排序（旧→新）
        files.sort()

        # 删除多余
        for f in files[:-Config.MAX_BACKUP_MODELS]:
            os.remove(os.path.join(backup_dir, f))