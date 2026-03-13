import cv2
import numpy as np
import mss


def find_all_rois():
    """
    這個輔助工具會引導你依序框選：
    1. P1 血量條
    2. P2 血量條
    3. P1 鬥氣條 (Drive Gauge)
    4. P2 鬥氣條 (Drive Gauge)
    """
    print("📸 準備擷取遊戲畫面...")

    # 擷取目前的畫面
    monitor = {"top": 0, "left": 0, "width": 1280, "height": 720}
    with mss.mss() as sct:
        sct_img = sct.grab(monitor)
        frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

    def select_and_format(window_name, prompt_text):
        print(f"\n--- {prompt_text} ---")
        print("請用滑鼠框選，確認請按【Enter】或【空白鍵】。")
        roi = cv2.selectROI(window_name, frame, showCrosshair=True)
        cv2.destroyWindow(window_name)

        # OpenCV 格式 (x, y, w, h) 轉 Numpy 切片格式 (y_start, y_end, x_start, x_end)
        x, y, w, h = roi
        return (y, y + h, x, x + w)

    # 依序執行 4 次框選
    p1_hp_roi = select_and_format("1. P1 Health Bar", "步驟 1: 框選 P1 (左方) 血量條")
    p2_hp_roi = select_and_format("2. P2 Health Bar", "步驟 2: 框選 P2 (右方) 血量條")
    p1_drive_roi = select_and_format("3. P1 Drive Gauge", "步驟 3: 框選 P1 (左方) 鬥氣條 (綠色方塊區)")
    p2_drive_roi = select_and_format("4. P2 Drive Gauge", "步驟 4: 框選 P2 (右方) 鬥氣條 (綠色方塊區)")

    cv2.destroyAllWindows()

    print("\n✅ 框選完成！請將以下程式碼複製，並覆蓋掉 vision_reader.py 裡面的 def __init__ 設定：\n")
    print("-" * 50)
    print(f"        self.p1_health_roi = {p1_hp_roi}")
    print(f"        self.p2_health_roi = {p2_hp_roi}")
    print(f"        self.p1_drive_roi = {p1_drive_roi}")
    print(f"        self.p2_drive_roi = {p2_drive_roi}")
    print("-" * 50)


if __name__ == "__main__":
    find_all_rois()