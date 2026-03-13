import cv2
import numpy as np
import mss
from ultralytics import YOLO


class VisionReader:
    def __init__(self, monitor_number=1):
        self.sct = mss.mss()
        self.monitor = {"top": 0, "left": 0, "width": 1280, "height": 720}

        self.p1_health_roi = (72, 100, 125, 594)
        self.p2_health_roi = (71, 99, 689, 1153)

        # 🌟 載入 YOLO 羅盤
        print("🚀 啟動 YOLOv8 視覺羅盤...")
        self.yolo_model = YOLO("yolov8n.pt")

        # 追蹤 P1 和 P2 的 X 座標 (預設 P1 在左邊，P2 在右邊)
        self.p1_x = 300
        self.p2_x = 900
        self.is_flipped = False

    # ... (保留你原本的 capture_frame 和 get_ai_observation 不變) ...

    def get_health_bars(self, frame):
        # ... (保留你原本的裁切與計算邏輯) ...
        # (這裡省略以節省版面，請保留你精準計算血量的程式碼)
        pass

    def update_positions(self, frame):
        """🌟 新增：使用 YOLO 更新人物座標，並判斷是否換位"""
        # 為了效能，我們可以把圖片縮小一點給 YOLO 看
        small_frame = cv2.resize(frame, (640, 360))
        results = self.yolo_model.predict(source=small_frame, classes=[0], conf=0.4, verbose=False)

        boxes = results[0].boxes
        current_x_centers = []

        for box in boxes:
            x1, _, x2, _ = map(int, box.xyxy[0])
            # 因為圖片縮小了一半，所以中心座標要乘 2 還原
            center_x = ((x1 + x2) / 2) * 2
            current_x_centers.append(center_x)

        # 如果剛好抓到兩個人
        if len(current_x_centers) >= 2:
            # 依照 X 座標由左至右排序
            current_x_centers.sort()
            left_person = current_x_centers[0]
            right_person = current_x_centers[1]

            # 使用距離判斷法：離上一次 P1 位置比較近的，就是 P1
            dist_p1_to_left = abs(self.p1_x - left_person)
            dist_p1_to_right = abs(self.p1_x - right_person)

            if dist_p1_to_left < dist_p1_to_right:
                # P1 在左邊 (正常狀態)
                self.p1_x = left_person
                self.p2_x = right_person
                self.is_flipped = False
            else:
                # P1 跑到右邊了 (換位狀態)
                self.p1_x = right_person
                self.p2_x = left_person
                self.is_flipped = True

        return self.is_flipped