import cv2
import numpy as np
import mss


class VisionReader:
    """
    負責擷取螢幕畫面、轉換 AI 觀察狀態，並解析精準的血量與鬥氣條區域。
    """

    def __init__(self, monitor_number=1):
        self.sct = mss.mss()
        self.monitor = {"top": 0, "left": 0, "width": 1280, "height": 720}

        # 使用你精準框選的座標
        self.p1_health_roi = (72, 100, 125, 594)
        self.p2_health_roi = (71, 99, 689, 1153)
        self.p1_drive_roi = (106, 120, 373, 593)
        self.p2_drive_roi = (106, 121, 690, 909)

    def capture_frame(self):
        """擷取當前螢幕畫面並轉換為 OpenCV 格式 (BGR)"""
        sct_img = self.sct.grab(self.monitor)
        frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
        return frame

    def get_ai_observation(self, frame):
        """轉換為 AI 適合的格式 (縮小 + 灰階)"""
        resized = cv2.resize(frame, (256, 144))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        observation = np.expand_dims(gray, axis=0)
        return observation

    def get_health_bars(self, frame):
        """回傳 P1 和 P2 的目前血量百分比"""
        p1_crop = frame[self.p1_health_roi[0]:self.p1_health_roi[1],
        self.p1_health_roi[2]:self.p1_health_roi[3]]
        p2_crop = frame[self.p2_health_roi[0]:self.p2_health_roi[1],
        self.p2_health_roi[2]:self.p2_health_roi[3]]

        p1_health = self._calculate_percentage(p1_crop)
        p2_health = self._calculate_percentage(p2_crop)
        return p1_health, p2_health

    def get_drive_bars(self, frame):
        """(未來擴充用) 回傳 P1 和 P2 的目前鬥氣條百分比"""
        p1_crop = frame[self.p1_drive_roi[0]:self.p1_drive_roi[1],
        self.p1_drive_roi[2]:self.p1_drive_roi[3]]
        p2_crop = frame[self.p2_drive_roi[0]:self.p2_drive_roi[1],
        self.p2_drive_roi[2]:self.p2_drive_roi[3]]

        p1_drive = self._calculate_percentage(p1_crop)
        p2_drive = self._calculate_percentage(p2_crop)
        return p1_drive, p2_drive

    def _calculate_percentage(self, crop_img):
        """內部函式：計算裁切圖片中的亮度比例 (血量/鬥氣量)"""
        gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray_crop, 100, 255, cv2.THRESH_BINARY)

        white_pixels = cv2.countNonZero(threshold)
        total_pixels = threshold.shape[0] * threshold.shape[1]
        return white_pixels / total_pixels


# ==========================================
# 測試程式碼
# ==========================================
if __name__ == "__main__":
    vision = VisionReader()
    frame = vision.capture_frame()
    p1_hp, p2_hp = vision.get_health_bars(frame)
    p1_drive, p2_drive = vision.get_drive_bars(frame)

    print(f"❤️  P1 血量: {p1_hp * 100:.2f}% | 🟩 P1 鬥氣: {p1_drive * 100:.2f}%")
    print(f"❤️  P2 血量: {p2_hp * 100:.2f}% | 🟩 P2 鬥氣: {p2_drive * 100:.2f}%")