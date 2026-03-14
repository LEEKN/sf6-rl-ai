import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

from vision_reader import VisionReader
from action_manager import SF6ActionManagerAsync


class SF6Env(gym.Env):
    def __init__(self, ai_side="P1", match_mode="training"):
        super(SF6Env, self).__init__()
        self.ai_side = ai_side
        self.match_mode = match_mode

        self.vision = VisionReader(debug_mode=True, ai_side=self.ai_side)
        self.action_manager = SF6ActionManagerAsync("sf6_moves.json")

        num_actions = len(self.action_manager.move_list)
        self.action_space = spaces.Discrete(num_actions)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(1, 144, 256), dtype=np.uint8
        )

        self.prev_my_health = 1.0
        self.prev_enemy_health = 1.0
        self.prev_my_drive = 1.0
        self.prev_enemy_drive = 1.0

        self.my_candidate = 1.0
        self.enemy_candidate = 1.0
        self.my_confirm_count = 0
        self.enemy_confirm_count = 0

        self.CONFIRM_STEPS = 3
        self.current_step = 0
        self.MAX_STEPS = 3000

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print("\n🛑 回合結束，觸發環境重置...")
        self.action_manager.reset_state()
        self.current_step = 0

        if self.match_mode == "versus":
            print("⏳ [時間凍結] 階段 1：等待退出當前對戰 (偵測血條消失)...")
            while True:
                frame = self.vision.capture_frame()
                left_h, right_h = self.vision.get_health_bars(frame)
                if left_h < 0.1 and right_h < 0.1:
                    print("🌑 確認進入非戰鬥畫面，大腦已暫停收集資料。")
                    break
                time.sleep(0.5)
            print("⏳ [時間凍結] 階段 2：尋找對手中，等待新戰局 (偵測滿血開局)...")
        else:
            print("⚡ [訓練模式] 準備無縫重置...")

        max_wait_time = 99999.0 if self.match_mode == "versus" else 20.0
        start_time = time.time()

        while True:
            frame = self.vision.capture_frame()
            left_health, right_health = self.vision.get_health_bars(frame)

            if left_health > 0.70 and right_health > 0.70:
                print(f"✅ 偵測到滿血，準備解除凍結！")
                self.vision.calibrate_max_values(frame)
                if self.match_mode == "versus":
                    print("⏳ [對戰模式] 等待 'FIGHT!' 動畫結束...")
                    time.sleep(2.0)
                actual_l_h, actual_r_h = self.vision.get_health_bars(frame)
                break

            if time.time() - start_time > max_wait_time:
                print("⚠️ 等待超時！強制開始新迴圈。")
                actual_l_h, actual_r_h = left_health, right_health
                break
            time.sleep(1.0)

        frame = self.vision.capture_frame()
        actual_l_d, actual_r_d = self.vision.get_drive_bars(frame)

        if self.ai_side == "P1":
            self.prev_my_health, self.prev_enemy_health = actual_l_h, actual_r_h
            self.prev_my_drive, self.prev_enemy_drive = actual_l_d, actual_r_d
        else:
            self.prev_my_health, self.prev_enemy_health = actual_r_h, actual_l_h
            self.prev_my_drive, self.prev_enemy_drive = actual_r_d, actual_l_d

        self.my_candidate = self.prev_my_health
        self.enemy_candidate = self.prev_enemy_health
        self.my_confirm_count = 0
        self.enemy_confirm_count = 0

        observation = self.vision.get_ai_observation(frame)
        print("▶️ [時間凍結解除] 新回合正式開始！")
        return observation, {}

    def step(self, action):
        self.current_step += 1
        action_id = int(action.item()) if hasattr(action, 'item') else int(action)
        reward = 0.0

        # ==========================================
        # 🌟 TODO 4：軟性動作遮罩與連打懲罰 (Soft Masking)
        # ==========================================
        # 檢查 Action Manager 目前是否還在處理上一招的硬直
        is_busy = self.action_manager.cooldown_frames > 0 or len(self.action_manager.macro_queue) > 0

        if is_busy and action_id != 0:
            # 如果 AI 在硬直期間試圖按按鍵，給予微小扣分，並強制轉為待機 (0)
            reward -= 0.05
            action_id = 0
            # 這裡不 print 避免洗頻，但 AI 的神經網路會默默學到教訓
        # ==========================================

        frame = self.vision.capture_frame()
        is_flipped = self.vision.update_positions(frame)

        # 將過濾後的 action_id 交給 Action Manager 執行
        self.action_manager.step(action_id, is_flipped=is_flipped)

        observation = self.vision.get_ai_observation(frame)
        left_h, right_h = self.vision.get_health_bars(frame)
        left_d, right_d = self.vision.get_drive_bars(frame)

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

        my_damage = self.prev_my_health - new_my_h
        enemy_damage = self.prev_enemy_health - new_enemy_h
        my_drive_loss = self.prev_my_drive - my_drive

        if enemy_damage > 0:
            reward += enemy_damage * 100
            print(f"🎉 傷害確認！獲得獎勵: +{enemy_damage * 100:.2f} (敵方剩餘: {new_enemy_h * 100:.1f}%)")

        if my_damage > 0:
            reward -= my_damage * 100
            print(f"⚠️ 受到真實傷害！扣除分數: -{my_damage * 100:.2f} (自己剩餘: {new_my_h * 100:.1f}%)")

        is_defending = action_id in [2, 3]
        if is_defending and my_drive_loss > 0.01 and my_damage <= 0.005:
            reward += 2.0
            print(f"🛡️ 成功防禦！獲得獎勵: +2.0 (消耗鬥氣: {my_drive_loss * 100:.1f}%)")

        terminated = bool(new_my_h <= 0.03 or new_enemy_h <= 0.03)
        if terminated:
            print(f"🛑 回合正式結束！(自己: {new_my_h * 100:.1f}% | 敵人: {new_enemy_h * 100:.1f}%)")

        truncated = bool(self.current_step >= self.MAX_STEPS)
        if truncated:
            print("⚠️ 回合時間過長，觸發環境強制重置 (Truncated)！")

        self.prev_my_health = new_my_h
        self.prev_enemy_health = new_enemy_h
        self.prev_my_drive = my_drive
        self.prev_enemy_drive = enemy_drive

        info = {}
        return observation, reward, terminated, truncated, info