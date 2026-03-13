import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

# 匯入我們之前寫好的模組
from vision_reader import VisionReader
from action_manager import SF6ActionManagerAsync


class SF6Env(gym.Env):
    """
    《快打旋風 6》的自定義 Gymnasium 環境。
    整合了視覺動態等待、自動校準血量、以及型別錯誤修復。
    """

    def __init__(self):
        super(SF6Env, self).__init__()

        self.vision = VisionReader()
        self.action_manager = SF6ActionManagerAsync("sf6_moves.json")

        num_actions = len(self.action_manager.move_list)
        self.action_space = spaces.Discrete(num_actions)

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(1, 144, 256),
            dtype=np.uint8
        )

        self.prev_p1_health = 1.0
        self.prev_p2_health = 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print("\n🛑 等待新回合開始 (等待雙方滿血)...")

        max_wait_time = 20.0
        start_time = time.time()

        actual_p1_full = 1.0
        actual_p2_full = 1.0

        while True:
            frame = self.vision.capture_frame()
            p1_health, p2_health = self.vision.get_health_bars(frame)

            print(f"   👁️ 視覺偵測中... P1: {p1_health * 100:.1f}% | P2: {p2_health * 100:.1f}%")

            # 滿血判定門檻 75%
            if p1_health > 0.75 and p2_health > 0.75:
                wait_duration = time.time() - start_time
                print(f"✅ 偵測到雙方滿血，新回合開始！(等待了 {wait_duration:.1f} 秒)\n")
                actual_p1_full = p1_health
                actual_p2_full = p2_health
                break

            if time.time() - start_time > max_wait_time:
                print("⚠️ 等待超時！強制開始新迴圈。")
                actual_p1_full = p1_health
                actual_p2_full = p2_health
                break

            time.sleep(1.0)

        # 紀錄真實滿血數值
        self.prev_p1_health = actual_p1_full
        self.prev_p2_health = actual_p2_full

        frame = self.vision.capture_frame()
        observation = self.vision.get_ai_observation(frame)

        return observation, {}

    def step(self, action):
        """
        遊戲迴圈的核心
        """
        # ⭐️ 關鍵修正 1：將 NumPy 陣列安全地轉換為 Python 整數
        if hasattr(action, 'item'):
            action_id = int(action.item())
        else:
            action_id = int(action)

        # 執行動作：傳入轉換好的 action_id
        self.action_manager.step(action_id)

        # 擷取新畫面與當前血量
        frame = self.vision.capture_frame()
        observation = self.vision.get_ai_observation(frame)
        p1_health, p2_health = self.vision.get_health_bars(frame)

        reward = 0.0
        p1_damage_taken = self.prev_p1_health - p1_health
        p2_damage_taken = self.prev_p2_health - p2_health

        if p2_damage_taken > 0:
            reward += p2_damage_taken * 100
            print(f"🎉 打中對手！獲得獎勵: +{p2_damage_taken * 100:.2f} (P2剩餘血量: {p2_health * 100:.1f}%)")

        if p1_damage_taken > 0:
            reward -= p1_damage_taken * 100
            print(f"⚠️ 受到傷害！扣除分數: -{p1_damage_taken * 100:.2f} (P1剩餘血量: {p1_health * 100:.1f}%)")

        # ⭐️ 關鍵修正 2：將死亡判定門檻提高到 10% (0.10)，避免背景雜訊干擾
        terminated = bool(p1_health < 0.03 or p2_health < 0.08)
        if terminated:
            print(f"🛑 回合結束！(血量歸零判定 - P1剩餘: {p1_health * 100:.1f}%, P2剩餘: {p2_health * 100:.1f}%)")

        self.prev_p1_health = p1_health
        self.prev_p2_health = p2_health

        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info