import pydirectinput
import time
import random


class KeyboardController:
    """
    這個類別負責將文字指令轉換為真實的底層鍵盤訊號送入遊戲。
    支援循序按鍵與精確的複合按鍵 (Chord) 控制。
    """

    def __init__(self):
        # 關閉 pydirectinput 預設的固定延遲，我們要用自己的精確控制
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
        執行單純的連續按鍵 (循序執行，非同時)
        """
        if not keys:
            return

        for key in keys:
            self.tap_key(key)
            between_keys_delay = random.uniform(0.01, 0.03)
            time.sleep(between_keys_delay)

    # ==========================================
    # 🌟 新增：支援複合按鍵 (Chord) 與精確時間控制
    # ==========================================
    def press_keys(self, keys):
        """
        同時按下多個按鍵 (例如 ['down', 'right'] 模擬斜下)
        """
        if not keys:
            return
        for key in keys:
            pydirectinput.keyDown(key)

    def release_keys(self, keys):
        """
        同時放開多個按鍵
        """
        if not keys:
            return
        for key in keys:
            pydirectinput.keyUp(key)


# ==========================================
# 測試程式碼
# ==========================================
if __name__ == "__main__":
    controller = KeyboardController()

    print("準備測試輸入，請在 3 秒內點擊任意一個文字編輯器或遊戲視窗...")
    time.sleep(3)

    print("🤖 模擬輸入: 單鍵輕拳 (q)")
    controller.execute_sequence(['q'])
    time.sleep(1)

    print("🤖 模擬輸入: 舊版循序輸入 (down -> right -> q)")
    controller.execute_sequence(['down', 'right', 'q'])
    time.sleep(1)

    print("🤖 模擬輸入: 新版精確複合輸入 (真實波動拳)")
    # 模擬 236 波動拳：下(按住) -> 下前(同時按住) -> 前+拳(同時按住)
    controller.press_keys(['down'])
    time.sleep(0.03)
    controller.press_keys(['right'])  # 此時 down 跟 right 都按著，等於斜下
    time.sleep(0.03)
    controller.release_keys(['down'])  # 放開 down，只剩 right
    controller.press_keys(['q'])  # 按下拳
    time.sleep(0.05)
    controller.release_keys(['right', 'q'])  # 全部放開

    print("✅ 輸入完成！我們現在擁有了精確控制按鍵疊加的能力。")