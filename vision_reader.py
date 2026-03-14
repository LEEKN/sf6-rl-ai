import cv2
import numpy as np
import mss
from ultralytics import YOLO


class VisionReader:
    # 🌟 新增 debug_mode 開關，預設為 False，要看畫面時可以改為 True
    def __init__(self, monitor_number=1, debug_mode=False, ai_side="P1"):  # 🌟 加入 ai_side
        self.sct = mss.mss()
        self.monitor = {"top": 0, "left": 0, "width": 1280, "height": 720}
        self.debug_mode = debug_mode
        self.ai_side = ai_side  # 紀錄 AI 陣營

        self.p1_health_roi = (76, 94, 124, 590)
        self.p2_health_roi = (76, 94, 685, 1153)
        self.p1_drive_roi = (105, 122, 371, 589)
        self.p2_drive_roi = (106, 119, 685, 903)

        print("🚀 啟動 YOLOv8 視覺羅盤...")
        self.yolo_model = YOLO("yolov8n.pt")

        # 🌟 根據 AI 陣營設定起始座標
        if self.ai_side == "P1":
            self.ai_x = 300  # P1 在左邊
            self.enemy_x = 900
        else:
            self.ai_x = 900  # P2 在右邊
            self.enemy_x = 300

        self.is_flipped = False
        # HSV 顏色過濾範圍設定 (針對舊版介面的高飽和度調整)
        self.health_lower = np.array([20, 100, 100])
        self.health_upper = np.array([40, 255, 255])

        self.drive_lower = np.array([45, 100, 100])
        self.drive_upper = np.array([85, 255, 255])

        # 🌟 動態校準系統的基準值
        self.is_calibrated = False
        self.max_px = {"p1_health": 1, "p2_health": 1, "p1_drive": 1, "p2_drive": 1}

    def capture_frame(self):
        sct_img = self.sct.grab(self.monitor)
        return cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

    def get_ai_observation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (256, 144))
        return np.expand_dims(resized, axis=0)

    # ==========================================
    # 動態校準系統
    # ==========================================
    def calibrate_max_values(self, frame):
        """🌟 在回合開始滿血時呼叫，將當下的像素數量定義為 100%"""
        p1_h_crop = frame[self.p1_health_roi[0]:self.p1_health_roi[1], self.p1_health_roi[2]:self.p1_health_roi[3]]
        p2_h_crop = frame[self.p2_health_roi[0]:self.p2_health_roi[1], self.p2_health_roi[2]:self.p2_health_roi[3]]
        p1_d_crop = frame[self.p1_drive_roi[0]:self.p1_drive_roi[1], self.p1_drive_roi[2]:self.p1_drive_roi[3]]
        p2_d_crop = frame[self.p2_drive_roi[0]:self.p2_drive_roi[1], self.p2_drive_roi[2]:self.p2_drive_roi[3]]

        self.max_px["p1_health"] = max(1, self._count_hsv_pixels(p1_h_crop, self.health_lower, self.health_upper))
        self.max_px["p2_health"] = max(1, self._count_hsv_pixels(p2_h_crop, self.health_lower, self.health_upper))
        self.max_px["p1_drive"] = max(1, self._count_hsv_pixels(p1_d_crop, self.drive_lower, self.drive_upper))
        self.max_px["p2_drive"] = max(1, self._count_hsv_pixels(p2_d_crop, self.drive_lower, self.drive_upper))

        self.is_calibrated = True
        print("🎯 視覺系統校準完成！已將當前血量與鬥氣鎖定為 100% 基準。")

    def _count_hsv_pixels(self, crop_img, lower_hsv, upper_hsv):
        """計算符合顏色的絕對像素數量"""
        hsv_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_crop, lower_hsv, upper_hsv)
        return cv2.countNonZero(mask)

    def _get_percentage(self, crop_img, lower_hsv, upper_hsv, max_key):
        """依照校準狀態，回傳精準的百分比"""
        pixels = self._count_hsv_pixels(crop_img, lower_hsv, upper_hsv)
        if self.is_calibrated:
            return min(1.0, pixels / self.max_px[max_key])  # 最高不超過 100%
        else:
            # 還沒校準前，暫時用面積當分母 (用來應付開局的滿血等待判定)
            total_pixels = crop_img.shape[0] * crop_img.shape[1]
            return pixels / total_pixels if total_pixels > 0 else 0.0

    def get_health_bars(self, frame):
        p1_crop = frame[self.p1_health_roi[0]:self.p1_health_roi[1], self.p1_health_roi[2]:self.p1_health_roi[3]]
        p2_crop = frame[self.p2_health_roi[0]:self.p2_health_roi[1], self.p2_health_roi[2]:self.p2_health_roi[3]]
        return (self._get_percentage(p1_crop, self.health_lower, self.health_upper, "p1_health"),
                self._get_percentage(p2_crop, self.health_lower, self.health_upper, "p2_health"))

    def get_drive_bars(self, frame):
        p1_crop = frame[self.p1_drive_roi[0]:self.p1_drive_roi[1], self.p1_drive_roi[2]:self.p1_drive_roi[3]]
        p2_crop = frame[self.p2_drive_roi[0]:self.p2_drive_roi[1], self.p2_drive_roi[2]:self.p2_drive_roi[3]]
        return (self._get_percentage(p1_crop, self.drive_lower, self.drive_upper, "p1_drive"),
                self._get_percentage(p2_crop, self.drive_lower, self.drive_upper, "p2_drive"))

    # ==========================================
    # YOLO 羅盤更新邏輯 (加入可視化)
    # ==========================================
    def update_positions(self, frame):
        small_frame = cv2.resize(frame, (640, 360))
        results = self.yolo_model.predict(source=small_frame, classes=[0], conf=0.4, verbose=False)

        boxes = results[0].boxes
        current_x_centers = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = ((x1 + x2) / 2) * 2
            current_x_centers.append(center_x)

            # Debug 模式：畫出 YOLO 辨識框
            if self.debug_mode:
                cv2.rectangle(small_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(small_frame, "Fighter", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if len(current_x_centers) >= 2:
            current_x_centers.sort()
            left_person = current_x_centers[0]
            right_person = current_x_centers[1]

            # 判斷左邊那個人是不是 AI
            dist_ai_to_left = abs(self.ai_x - left_person)
            dist_ai_to_right = abs(self.ai_x - right_person)

            if dist_ai_to_left < dist_ai_to_right:
                # AI 在左邊
                self.ai_x = left_person
                self.enemy_x = right_person
                # 如果我們是 P1，在左邊就是沒換位；如果是 P2，在左邊就是換位了！
                self.is_flipped = False if self.ai_side == "P1" else True
            else:
                # AI 在右邊
                self.ai_x = right_person
                self.enemy_x = left_person
                # 如果我們是 P1，在右邊就是換位了；如果是 P2，在右邊是正常的！
                self.is_flipped = True if self.ai_side == "P1" else False
        # 🌟 顯示 Debug 視窗
        if self.debug_mode:
            direction = "FLIPPED!" if self.is_flipped else "Normal"
            cv2.putText(small_frame, f"Status: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("YOLO Debug Window", small_frame)
            cv2.waitKey(1)

        return self.is_flipped