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

        # ==========================================
        # 🌟 TODO 2：升級為多模態觀察空間 (Dict)
        # ==========================================
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(1, 144, 256), dtype=np.uint8),
            # [自己血量, 敵人血量, 自己鬥氣, 敵人鬥氣, 冷卻倒數, 幀數優劣, 敵方硬直]
            "stats": spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        })

        # 新增追蹤變數
        self.current_frame_adv = 0.0
        self.enemy_stun_frames = 0.0
        self.last_action_id = 0
        # ==========================================

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

        # TODO 7: 新增單局數據追蹤
        self.episode_damage_dealt = 0.0
        self.episode_damage_taken = 0.0

        # TODO 8 防禦疲勞追蹤器
        self.consecutive_defends = 0

        # TODO 18: 新增狀態推演追蹤變數
        self.current_frame_adv = 0.0
        self.enemy_stun_frames = 0.0
        self.last_action_id = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print("\n🛑 回合結束，觸發環境重置...")
        self.action_manager.reset_state()
        self.current_step = 0

        # TODO 7: 新回合數據歸零
        self.episode_damage_dealt = 0.0
        self.episode_damage_taken = 0.0
        self.consecutive_defends = 0

        # TODO 18: 新回合重置幀數狀態
        self.current_frame_adv = 0.0
        self.enemy_stun_frames = 0.0
        self.last_action_id = 0

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

        # TODO 18: 打包成 7 維 Dict 格式 (初始值冷卻和幀數皆為 0)
        obs_dict = {
            "image": observation,
            "stats": np.array([
                self.prev_my_health, self.prev_enemy_health,
                self.prev_my_drive, self.prev_enemy_drive,
                0.0, 0.0, 0.0  # 冷卻倒數, 幀數優劣, 敵方硬直
            ], dtype=np.float32)
        }

        print("▶️ [時間凍結解除] 新回合正式開始！")
        return obs_dict, {}

    def step(self, action):
        self.current_step += 1
        action_id = int(action.item()) if hasattr(action, 'item') else int(action)
        reward = 0.0

        is_busy = self.action_manager.cooldown_frames > 0 or len(self.action_manager.macro_queue) > 0
        if is_busy and action_id != 0:
            reward -= 0.05
            action_id = 0

        frame = self.vision.capture_frame()
        is_flipped = self.vision.update_positions(frame)
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
            self.episode_damage_dealt += enemy_damage  # 🌟 記錄造成傷害
            print(f"🎉 傷害確認！獲得獎勵: +{enemy_damage * 100:.2f} (敵方剩餘: {new_enemy_h * 100:.1f}%)")

        if my_damage > 0:
            reward -= my_damage * 100
            self.episode_damage_taken += my_damage  # 🌟 記錄承受傷害
            print(f"⚠️ 受到真實傷害！扣除分數: -{my_damage * 100:.2f} (自己剩餘: {new_my_h * 100:.1f}%)")

        is_defending = action_id in [2, 3]  # 後退防禦 或 下蹲防禦
        is_moving_forward = action_id == 1  # 前進

        if is_defending:
            if my_drive_loss > 0.01 and my_damage <= 0.005:
                # 成功防禦，增加疲勞度
                self.consecutive_defends += 1

                # 計算遞減獎勵 (從 2.0 開始扣除，最低降到 0.1)
                defense_reward = max(0.1, 2.0 - (self.consecutive_defends * 0.4))
                reward += defense_reward
                print(f"🛡️ 成功防禦！獲得獎勵: +{defense_reward:.2f} (連續防禦計數: {self.consecutive_defends})")
        else:
            # 只要 AI 放棄防禦做其他動作，疲勞值歸零
            self.consecutive_defends = 0

        if is_moving_forward:
            # 給予微小的推進獎勵，鼓勵主動拉近距離 (對抗龜縮)
            reward += 0.05

        # ==========================================
        # TODO 18: 內部狀態機推演 (Frame Advantage & Stun)
        # ==========================================
        last_move = self.action_manager.move_list.get(self.last_action_id, {})

        # 當前一招剛好脫離硬直的瞬間，結算該招的結果
        if self.last_action_id != 0 and not is_busy:
            if enemy_damage > 0:
                # 命中 (Hit)
                adv = last_move.get("on_hit_adv", 0)
                stun = last_move.get("hitstun", 0)
                self.current_frame_adv = max(-60, min(60, adv))
                self.enemy_stun_frames = stun
            elif self.prev_enemy_drive - enemy_drive > 0.01 and enemy_damage <= 0:
                # 被防禦 (Block)
                adv = last_move.get("on_block_adv", 0)
                stun = last_move.get("hitstun", 0) * 0.6  # 估算防禦硬直較短
                self.current_frame_adv = max(-60, min(60, adv))
                self.enemy_stun_frames = stun

        # 時間流逝 (扣除幀數優劣與硬直)
        time_passed = self.action_manager.frames_per_step
        if self.current_frame_adv > 0:
            self.current_frame_adv = max(0, self.current_frame_adv - time_passed)
        elif self.current_frame_adv < 0:
            self.current_frame_adv = min(0, self.current_frame_adv + time_passed)

        self.enemy_stun_frames = max(0, self.enemy_stun_frames - time_passed)

        # 更新最後一次的有效動作 (如果是待機就不更新)
        if action_id != 0:
            self.last_action_id = action_id

        # 取得冷卻倒數
        cooldown_norm = max(0, self.action_manager.cooldown_frames) / 60.0

        # 正規化幀數 (-1.0 ~ 1.0)
        norm_adv = self.current_frame_adv / 60.0
        norm_stun = self.enemy_stun_frames / 60.0

        terminated = bool(new_my_h <= 0.03 or new_enemy_h <= 0.03)
        info = {}  # 準備 info 字典

        if terminated:
            print(f"🛑 回合正式結束！(自己: {new_my_h * 100:.1f}% | 敵人: {new_enemy_h * 100:.1f}%)")
            # 🌟 TODO 7: 回合結束時，結算戰績並放入 info
            info["episode_damage_dealt"] = self.episode_damage_dealt
            info["episode_damage_taken"] = self.episode_damage_taken
            # 判斷輸贏：如果敵人血量先到底，算贏 (1.0)，否則算輸 (0.0)
            info["win"] = 1.0 if new_enemy_h <= 0.03 and new_my_h > 0.03 else 0.0
            info["match_length"] = self.current_step

        truncated = bool(self.current_step >= self.MAX_STEPS)
        if truncated:
            print("⚠️ 回合時間過長，觸發環境強制重置 (Truncated)！")

        # TODO 18: 將推演結果打包送給大腦
        obs_dict = {
            "image": observation,
            "stats": np.array([
                new_my_h, new_enemy_h,
                my_drive, enemy_drive,
                cooldown_norm, norm_adv, norm_stun
            ], dtype=np.float32)
        }

        self.prev_my_health = new_my_h
        self.prev_enemy_health = new_enemy_h
        self.prev_my_drive = my_drive
        self.prev_enemy_drive = enemy_drive

        return obs_dict, reward, terminated, truncated, info