import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

from vision_reader import VisionReader
from action_manager import SF6ActionManagerAsync


class SF6Env(gym.Env):
    """
    《快打旋風 6》的自定義 Gymnasium 環境。
    支援 P1/P2 動態陣營切換與 My/Enemy 獎勵解耦。
    """

    # 🌟 修正 1：加入了 ai_side 和 match_mode 參數
    def __init__(self, ai_side="P1", match_mode="training"):
        super(SF6Env, self).__init__()

        self.ai_side = ai_side
        self.match_mode = match_mode

        self.vision = VisionReader(debug_mode=True, ai_side=self.ai_side)
        self.action_manager = SF6ActionManagerAsync("sf6_moves.json")

        num_actions = len(self.action_manager.move_list)
        self.action_space = spaces.Discrete(num_actions)

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(1, 144, 256),
            dtype=np.uint8
        )

        # 🌟 修正 2：徹底改用 My / Enemy 變數
        self.prev_my_health = 1.0
        self.prev_enemy_health = 1.0
        self.my_candidate = 1.0
        self.enemy_candidate = 1.0
        self.my_confirm_count = 0
        self.enemy_confirm_count = 0

        self.CONFIRM_STEPS = 3
        self.current_step = 0
        self.MAX_STEPS = 3000

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print("\n🛑 等待新回合開始 (等待雙方滿血)...")

        self.action_manager.reset_state()
        self.current_step = 0

        max_wait_time = 20.0
        start_time = time.time()

        while True:
            frame = self.vision.capture_frame()
            left_health, right_health = self.vision.get_health_bars(frame)

            if left_health > 0.70 and right_health > 0.70:
                print(f"✅ 偵測到雙方滿血，準備開始！")
                self.vision.calibrate_max_values(frame)

                if self.match_mode == "versus":
                    print("⏳ [對戰模式] 等待 'FIGHT!' 動畫結束...")
                    time.sleep(2.0)
                else:
                    print("⚡ [訓練模式] 無縫開戰！")

                actual_l, actual_r = self.vision.get_health_bars(frame)
                break

            if time.time() - start_time > max_wait_time:
                print("⚠️ 等待超時！強制開始新迴圈。")
                actual_l, actual_r = left_health, right_health
                break

            time.sleep(1.0)

        # 🌟 修正 3：依照設定的陣營，把左右血量分配給 My 和 Enemy
        if self.ai_side == "P1":
            self.prev_my_health, self.prev_enemy_health = actual_l, actual_r
        else:
            self.prev_my_health, self.prev_enemy_health = actual_r, actual_l

        self.my_candidate = self.prev_my_health
        self.enemy_candidate = self.prev_enemy_health
        self.my_confirm_count = 0
        self.enemy_confirm_count = 0

        frame = self.vision.capture_frame()
        observation = self.vision.get_ai_observation(frame)

        return observation, {}

    def step(self, action):
        self.current_step += 1

        if hasattr(action, 'item'):
            action_id = int(action.item())
        else:
            action_id = int(action)

        frame = self.vision.capture_frame()
        is_flipped = self.vision.update_positions(frame)
        self.action_manager.step(action_id, is_flipped=is_flipped)

        observation = self.vision.get_ai_observation(frame)
        left_h, right_h = self.vision.get_health_bars(frame)
        left_d, right_d = self.vision.get_drive_bars(frame)

        # 🌟 修正 4：依照設定的陣營，抓取對應的血量與鬥氣
        if self.ai_side == "P1":
            raw_my_h, raw_enemy_h = left_h, right_h
            my_drive, enemy_drive = left_d, right_d
        else:
            raw_my_h, raw_enemy_h = right_h, left_h
            my_drive, enemy_drive = right_d, left_d

        def verify_health(raw, stable, candidate, count):
            if raw > stable + 0.05 or raw < stable - 0.65:
                return stable, stable, 0
            if raw < stable - 0.015:
                if abs(raw - candidate) < 0.03:
                    count += 1
                else:
                    candidate, count = raw, 1
            else:
                count = 0
            if count >= self.CONFIRM_STEPS:
                stable, count = candidate, 0
            return stable, candidate, count

        new_my_h, self.my_candidate, self.my_confirm_count = verify_health(
            raw_my_h, self.prev_my_health, self.my_candidate, self.my_confirm_count)

        new_enemy_h, self.enemy_candidate, self.enemy_confirm_count = verify_health(
            raw_enemy_h, self.prev_enemy_health, self.enemy_candidate, self.enemy_confirm_count)

        # 🌟 修正 5：獎勵邏輯解耦 (不管在左在右，永遠是敵人扣血加分，自己扣血扣分)
        reward = 0.0
        my_damage = self.prev_my_health - new_my_h
        enemy_damage = self.prev_enemy_health - new_enemy_h

        if enemy_damage > 0:
            reward += enemy_damage * 100
            print(
                f"🎉 傷害確認！獲得獎勵: +{enemy_damage * 100:.2f} (敵方剩餘: {new_enemy_h * 100:.1f}%, 鬥氣: {enemy_drive * 100:.1f}%)")

        if my_damage > 0:
            reward -= my_damage * 100
            print(
                f"⚠️ 受到真實傷害！扣除分數: -{my_damage * 100:.2f} (自己剩餘: {new_my_h * 100:.1f}%, 鬥氣: {my_drive * 100:.1f}%)")

        # 🌟 修正 6：清理重複的死亡判定
        terminated = bool(new_my_h <= 0.03 or new_enemy_h <= 0.03)
        if terminated:
            print(
                f"🛑 回合正式結束！(自己: {new_my_h * 100:.1f}% 鬥氣 {my_drive * 100:.1f}% | 敵人: {new_enemy_h * 100:.1f}% 鬥氣 {enemy_drive * 100:.1f}%)")

        truncated = bool(self.current_step >= self.MAX_STEPS)
        if truncated:
            print("⚠️ 回合時間過長，觸發環境強制重置 (Truncated)！")

        self.prev_my_health = new_my_h
        self.prev_enemy_health = new_enemy_h

        info = {}
        return observation, reward, terminated, truncated, info