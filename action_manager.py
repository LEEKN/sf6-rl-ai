import json
import os
from keyboard_input import KeyboardController

class SF6ActionManagerAsync:
    """
    資料驅動的非阻塞動作管理員 (支援巨集佇列系統)。
    """
    def __init__(self, config_file="sf6_moves.json"):
        self.frames_per_step = 4
        self.cooldown_frames = 0
        self.config_file = config_file

        self.move_list = self._load_moves()
        self.keyboard = KeyboardController()

        # 巨集佇列，用來暫存還沒打完的連續技步驟
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
                self._execute_action(next_step['keys'], next_step['frames'], "巨集連段進行中...", is_flipped=is_flipped)
                action_executed = True

            else:
                # 取得 AI 決定的新招式，預設為待機 (0)
                move = self.move_list.get(action_id, self.move_list[0])

                # 🌟 判斷：這是一般招式還是巨集？
                if move.get("is_macro", False):
                    print(f"🔥 [啟動巨集] {move['name']}")
                    # 把巨集動作複製進佇列
                    self.macro_queue = move["sequence"].copy()

                    # 立即打出第一發
                    first_step = self.macro_queue.pop(0)
                    self._execute_action(first_step['keys'], first_step['frames'], "巨集起手", is_flipped=is_flipped)

                else:
                    # 一般單發招式
                    if action_id != 0:
                        self._execute_action(move['keys'], move['frames'], move['name'], is_flipped=is_flipped)
                    else:
                        # 待機不按鍵，只消耗 1 幀
                        self.cooldown_frames = move['frames']

                action_executed = True

        # 推進時間
        self.cooldown_frames -= self.frames_per_step
        return action_executed

    def _execute_action(self, keys, frames, debug_name, is_flipped=False):
        """內部函式：負責送出按鍵並設定冷卻硬直"""
        # 核心翻轉邏輯
        final_keys = []
        if is_flipped:
            for key in keys:
                if key == 'left':
                    final_keys.append('right')
                elif key == 'right':
                    final_keys.append('left')
                else:
                    final_keys.append(key)
        else:
            final_keys = keys

        flip_tag = "[換位翻轉] " if is_flipped and final_keys != keys else ""
        print(f"   💥 {flip_tag}[{debug_name}] 送出按鍵: {final_keys} | 消耗幀數: {frames} 幀")

        self._press_keys(final_keys)
        self.cooldown_frames = frames

    def _press_keys(self, keys):
        if not keys:
            return
        self.keyboard.execute_sequence(keys)

# ==========================================
# 測試程式碼
# ==========================================
if __name__ == "__main__":
    import time

    manager = SF6ActionManagerAsync("sf6_moves.json")

    print("\n--- 測試巨集連段 (加入真實時間模擬) ---")
    manager.cooldown_frames = 0
    manager.step(10)  # 啟動「下中腳確認波動拳」巨集

    # 模擬遊戲迴圈：假設遊戲是 60 FPS，每個 step 我們設定推進 4 幀
    # 4 幀的時間大約是 4 / 60 = 0.066 秒
    for i in range(15):
        manager.step(0)  # AI 嘗試閒置
        time.sleep(4 / 60.0)  # 讓程式真的睡一下，模擬遊戲時間流逝