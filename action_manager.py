import json
import os
from keyboard_input import KeyboardController

class SF6ActionManagerAsync:
    """
    資料驅動的非阻塞動作管理員。
    透過讀取外部 JSON 檔案來獲取招式幀數，輕鬆應對遊戲版本更新。
    """
    def __init__(self, config_file="sf6_moves.json"):
        self.frames_per_step = 4 
        self.cooldown_frames = 0
        self.config_file = config_file
        
        # 啟動時自動載入外部招式設定檔
        self.move_list = self._load_moves()
        self.keyboard = KeyboardController()

    def _load_moves(self):
        """讀取 JSON 設定檔。如果檔案不存在，會給出清楚的錯誤提示。"""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"❌ 找不到招式設定檔 {self.config_file}！請確認檔案是否存在。")
            
        with open(self.config_file, 'r', encoding='utf-8') as f:
            # 將 JSON 內容載入為 Python 字典
            # 注意：JSON 的 key 會是字串 (例如 "1")，我們需要轉回整數 (1) 方便 AI 使用
            raw_data = json.load(f)
            parsed_moves = {}
            for key, value in raw_data.items():
                parsed_moves[int(key)] = value
                
        print(f"✅ 成功載入 {len(parsed_moves)} 個招式設定！")
        return parsed_moves

    def step(self, action_id):
        """執行 AI 指令並管理冷卻時間"""
        action_executed = False
        
        if self.cooldown_frames <= 0:
            # 取得招式，預設為待機 (0)
            move = self.move_list.get(action_id, self.move_list[0])
            
            if action_id != 0:
                print(f"💥 [出招] {move['name']} | 消耗幀數: {move['frames']} 幀")
                self._press_keys(move['keys'])
            
            self.cooldown_frames = move['frames']
            action_executed = True
        
        # 推進時間
        self.cooldown_frames -= self.frames_per_step
        return action_executed

    def _press_keys(self, keys):
        """內部函式：未來串接真實鍵盤 API"""
        if not keys:
            return
        print(f"   ⌨️ [模擬鍵盤] 送出按鍵: {keys}")

        self.keyboard.execute_sequence(keys)

# ==========================================
# 測試程式碼
# ==========================================
if __name__ == "__main__":
    manager = SF6ActionManagerAsync("sf6_moves.json")
    manager.step(3) # 測試波動拳j