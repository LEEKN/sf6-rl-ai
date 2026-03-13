import pydirectinput
import time
import random


class KeyboardController:
    """
    這個類別負責將文字指令 (例如 'j', 'down') 轉換為真實的底層鍵盤訊號送入遊戲。
    並加入隨機延遲，模擬真實人類玩家的按鍵習慣，避免被偵測為外掛。
    """

    def __init__(self):
        # 關閉 pydirectinput 預設的固定延遲，我們要用自己的「隨機延遲」來取代
        pydirectinput.PAUSE = 0

    def tap_key(self, key):
        """
        模擬人類敲擊單一按鍵 (按下 -> 隨機微小延遲 -> 放開)
        """
        pydirectinput.keyDown(key)

        # 模擬人類按壓按鍵的持續時間 (約 0.02 到 0.06 秒之間隨機)
        press_duration = random.uniform(0.02, 0.06)
        time.sleep(press_duration)

        pydirectinput.keyUp(key)

    def execute_sequence(self, keys):
        """
        執行連續按鍵 (例如波動拳的 ['down', 'right', 'j'])
        """
        if not keys:
            return

        for key in keys:
            self.tap_key(key)

            # 按鍵與按鍵之間的移動間隔 (約 0.01 到 0.03 秒之間隨機)
            # 例如從「下」滑動到「前」需要一點點時間
            between_keys_delay = random.uniform(0.01, 0.03)
            time.sleep(between_keys_delay)


# ==========================================
# 測試程式碼
# ==========================================
if __name__ == "__main__":
    controller = KeyboardController()

    print("準備測試輸入，請在 3 秒內點擊任意一個文字編輯器或遊戲視窗...")
    time.sleep(3)

    print("🤖 模擬輸入: 輕拳 (j)")
    controller.execute_sequence(['j'])
    time.sleep(1)

    print("🤖 模擬輸入: 波動拳 (down -> right -> j)")
    controller.execute_sequence(['down', 'right', 'j'])
    print("✅ 輸入完成！你可以觀察每次按鍵的感覺是否像人類。")