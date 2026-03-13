import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

from vision_reader import VisionReader
from action_manager import SF6ActionManagerAsync


class SF6Env(gym.Env):
    """
    《快打旋風 6》的自定義 Gymnasium 環境。
    搭載了應對 SA3 的「延遲確認系統」與「3% 精準空血判定」。
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

        # 延遲確認系統變數
        self.p1_candidate = 1.0
        self.p2_candidate = 1.0
        self.p1_confirm_count = 0
        self.p2_confirm_count = 0
        self.CONFIRM_STEPS = 3

        # 🌟 新增：環境超時保險絲 (防止無限迴圈)
        self.current_step = 0
        self.MAX_STEPS = 3000  # 約等於遊戲時間 3 分鐘

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print("\n🛑 等待新回合開始 (等待雙方滿血)...")

        self.action_manager.reset_state()
        self.current_step = 0  # 🌟 重置步數計時器

        max_wait_time = 20.0
        start_time = time.time()

        actual_p1_full = 1.0
        actual_p2_full = 1.0

        while True:
            frame = self.vision.capture_frame()
            p1_health, p2_health = self.vision.get_health_bars(frame)

            print(f"   👁️ 視覺偵測中... P1: {p1_health * 100:.1f}% | P2: {p2_health * 100:.1f}%")

            if p1_health > 0.70 and p2_health > 0.70:
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

        self.prev_p1_health = actual_p1_full
        self.prev_p2_health = actual_p2_full
        self.p1_candidate = actual_p1_full
        self.p2_candidate = actual_p2_full
        self.p1_confirm_count = 0
        self.p2_confirm_count = 0

        frame = self.vision.capture_frame()
        observation = self.vision.get_ai_observation(frame)

        return observation, {}

    def step(self, action):
        self.current_step += 1  # 推進計時器

        if hasattr(action, 'item'):
            action_id = int(action.item())
        else:
            action_id = int(action)

        frame = self.vision.capture_frame()
        is_flipped = self.vision.update_positions(frame)

        # 將方位情報傳給動作管理員
        self.action_manager.step(action_id, is_flipped=is_flipped)

        # 取得 AI 觀察值與血量 (因為上面已經擷取過 frame 了，直接沿用即可)
        observation = self.vision.get_ai_observation(frame)
        raw_p1, raw_p2 = self.vision.get_health_bars(frame)

        def verify_health(raw, stable, candidate, count):
            # 物理極限濾波 (包容老桑 CA 65% 傷害)
            if raw > stable + 0.05 or raw < stable - 0.65:
                return stable, stable, 0

                # 進入確認期
            if raw < stable - 0.015:
                if abs(raw - candidate) < 0.03:
                    count += 1
                else:
                    candidate = raw
                    count = 1
            else:
                count = 0

            # 蓋章確認
            if count >= self.CONFIRM_STEPS:
                stable = candidate
                count = 0

            return stable, candidate, count

        new_p1, self.p1_candidate, self.p1_confirm_count = verify_health(
            raw_p1, self.prev_p1_health, self.p1_candidate, self.p1_confirm_count)

        new_p2, self.p2_candidate, self.p2_confirm_count = verify_health(
            raw_p2, self.prev_p2_health, self.p2_candidate, self.p2_confirm_count)

        # 計算獎勵
        reward = 0.0
        p1_damage = self.prev_p1_health - new_p1
        p2_damage = self.prev_p2_health - new_p2

        if p2_damage > 0:
            reward += p2_damage * 100
            print(f"🎉 傷害確認！獲得獎勵: +{p2_damage * 100:.2f} (P2剩餘: {new_p2 * 100:.1f}%)")

        if p1_damage > 0:
            reward -= p1_damage * 100
            print(f"⚠️ 受到真實傷害！扣除分數: -{p1_damage * 100:.2f} (P1剩餘: {new_p1 * 100:.1f}%)")

        # ==========================================
        # 🌟 死亡與超時判定
        # ==========================================
        # ⭐️ 關鍵修正：將死亡門檻提高到 3% (0.03)，完美涵蓋血槽外框的視覺殘留！
        terminated = bool(new_p1 <= 0.03 or new_p2 <= 0.02)
        if terminated:
            print(f"🛑 回合正式結束！(血量低於 3% 判定 K.O. - P1: {new_p1 * 100:.1f}%, P2: {new_p2 * 100:.1f}%)")

        # ⭐️ 新增保險：如果這局打太久卡住了，強制截斷 (Truncated) 進入下一回合
        truncated = bool(self.current_step >= self.MAX_STEPS)
        if truncated:
            print("⚠️ 回合時間過長，觸發環境強制重置 (Truncated)！")

        self.prev_p1_health = new_p1
        self.prev_p2_health = new_p2

        info = {}

        return observation, reward, terminated, truncated, info