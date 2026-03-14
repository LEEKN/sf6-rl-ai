import json
import os
import time
import random
from keyboard_input import KeyboardController


class SF6ActionManagerAsync:
    """
    資料驅動的非阻塞動作管理員 (支援巨集佇列系統與精準複合指令)。
    """

    def __init__(self, config_file="sf6_moves.json"):
        self.frames_per_step = 4
        self.cooldown_frames = 0
        self.config_file = config_file

        self.move_list = self._load_moves()
        self.keyboard = KeyboardController()

        # 巨集與複合指令佇列，用來暫存還沒打完的連續技步驟
        self.macro_queue = []

    # 在回合結束時，切斷 AI 正在按的按鍵與巨集
    def reset_state(self):
        """強制清空當前動作與冷卻，避免把這回合的連段帶到下一回合"""
        self.macro_queue.clear()
        self.cooldown_frames = 0
        print("🧹 [Action Manager] 已強制清空巨集佇列與冷卻狀態。")

    def _load_moves(self):
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"❌ 找不到招式設定檔 {self.config_file}！請確認檔案是否存在。")

        with open(self.config_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            parsed_moves = {}
            for key, value in raw_data.items():
                parsed_moves[int(key)] = value

        print(f"✅ 成功載入 {len(parsed_moves)} 個招式與巨集設定！")
        return parsed_moves

    def step(self, action_id, is_flipped=False):
        """執行 AI 指令，管理冷卻時間與巨集佇列"""
        action_executed = False

        # 只有在沒有冷卻硬直時，才允許出下一招
        if self.cooldown_frames <= 0:

            # 🌟 優先處理：如果巨集還沒打完，繼續打下一步
            if len(self.macro_queue) > 0:
                next_step = self.macro_queue.pop(0)

                # 支援巨集直接呼叫其他複雜招式 (例如巨集裡直接放波動拳的 ID)
                if "id" in next_step:
                    move = self.move_list.get(next_step["id"])
                    if move:
                        self._process_move(move, is_flipped)
                else:
                    self._execute_action(next_step['keys'], next_step['frames'], "巨集連段進行中...",
                                         is_flipped=is_flipped)

                action_executed = True

            else:
                # 取得 AI 決定的新招式，預設為待機 (0)
                move = self.move_list.get(action_id, self.move_list[0])
                self._process_move(move, is_flipped)
                action_executed = True

        # 推進時間
        self.cooldown_frames -= self.frames_per_step
        return action_executed

    def _process_move(self, move, is_flipped):
        """內部函式：判斷並分流處理巨集、複合招式(必殺技)、或一般招式"""
        if move.get("is_macro", False):
            print(f"🔥 [啟動巨集] {move['name']}")
            # 把巨集動作複製進佇列
            self.macro_queue = move["sequence"].copy()

            # 立即打出第一發
            first_step = self.macro_queue.pop(0)
            if "id" in first_step:
                self._process_move(self.move_list[first_step["id"]], is_flipped)
            else:
                self._execute_action(first_step['keys'], first_step['frames'], "巨集起手", is_flipped=is_flipped)

        elif move.get("is_complex", False):
            # 🌟 處理精確微操作的必殺技
            self._execute_complex_action(move['inputs'], move['frames'], move['name'], is_flipped)

        else:
            # 一般單發招式
            if move["name"] != "待機 (Idle)":
                self._execute_action(move['keys'], move['frames'], move['name'], is_flipped=is_flipped)
            else:
                # 待機不按鍵，只消耗 1 幀
                self.cooldown_frames = move['frames']

    def _flip_keys(self, keys, is_flipped):
        """內部函式：處理 P1/P2 左右換位的按鍵翻轉"""
        if not is_flipped:
            return keys

        flipped_keys = []
        for key in keys:
            if key == 'left':
                flipped_keys.append('right')
            elif key == 'right':
                flipped_keys.append('left')
            else:
                flipped_keys.append(key)
        return flipped_keys

    def _execute_action(self, keys, frames, debug_name, is_flipped=False):
        """內部函式：一般單發按鍵送出 (快速點擊)，加入隨機按壓時間防偵測"""
        final_keys = self._flip_keys(keys, is_flipped)
        flip_tag = "[換位翻轉] " if is_flipped and final_keys != keys else ""
        print(f"   💥 {flip_tag}[{debug_name}] 送出按鍵: {final_keys} | 消耗幀數: {frames} 幀")

        if final_keys:
            self.keyboard.press_keys(final_keys)

            # 🌟 加入隨機延遲 (模擬單次按鍵的按壓時間，約 0.02 到 0.05 秒)
            press_duration = random.uniform(0.02, 0.05)
            time.sleep(press_duration)

            self.keyboard.release_keys(final_keys)

        self.cooldown_frames = frames

    def _execute_complex_action(self, inputs, frames, debug_name, is_flipped=False):
        """內部函式：精確執行複合必殺技，並在每一步加入隨機抖動 (Jitter) 防偵測"""
        print(f"   🌪️ [{debug_name}] 啟動複合指令精確輸入 (含防偵測隨機抖動)...")

        for step in inputs:
            final_keys = self._flip_keys(step["press"], is_flipped)

            # 從 JSON 取得基準等待時間，預設 0.03 秒
            base_wait = step.get("wait", 0.03)

            # 🌟 產生一個微小的隨機誤差 (例如 +/- 0.008 秒)
            jitter = random.uniform(-0.008, 0.008)

            # 計算真實等待時間，並確保不能低於一個極限值 (避免按太快遊戲吃不到)
            actual_wait = max(0.015, base_wait + jitter)

            if final_keys:
                self.keyboard.press_keys(final_keys)

            # 睡眠真實的隨機時間
            time.sleep(actual_wait)

            if final_keys:
                self.keyboard.release_keys(final_keys)

        self.cooldown_frames = frames

# ==========================================
# 測試程式碼
# ==========================================
if __name__ == "__main__":
    manager = SF6ActionManagerAsync("sf6_moves.json")

    print("\n--- 測試巨集連段與複合指令 (加入真實時間模擬) ---")
    manager.cooldown_frames = 0

    # 測試啟動「下中腳確認波動拳」巨集 (假設 JSON 中此招式的 ID 為 23)
    # 若你還沒更新 sf6_moves.json，請確認裡面有沒有 ID 23 的設定！
    manager.step(23)

    # 模擬遊戲迴圈：假設遊戲是 60 FPS，每個 step 我們設定推進 4 幀
    # 4 幀的時間大約是 4 / 60 = 0.066 秒
    for i in range(20):
        manager.step(0)  # AI 嘗試閒置
        time.sleep(4 / 60.0)  # 讓程式真的睡一下，模擬遊戲時間流逝