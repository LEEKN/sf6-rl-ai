import cv2
import mss
import numpy as np
from ultralytics import YOLO


def test_yolo_tracking():
    print("🚀 載入 YOLOv8 模型中...")
    # 第一次執行時，會自動下載 yolov8n.pt (約 6MB 的輕量化模型)
    model = YOLO("yolov8n.pt")

    sct = mss.mss()
    monitor = {"top": 0, "left": 0, "width": 1280, "height": 720}

    print("✅ 載入完成！請開啟《快打旋風 6》並保持在左上角。")
    print("按下 'q' 鍵可結束測試。")

    while True:
        # 1. 擷取畫面
        sct_img = sct.grab(monitor)
        frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

        # 2. 讓 YOLO 進行辨識 (只辨識 classes=[0]，也就是 'person' 人類)
        results = model.predict(source=frame, classes=[0], conf=0.4, verbose=False)

        # 3. 解析結果
        boxes = results[0].boxes

        # 把辨識框畫到畫面上
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # 畫框與標示信心度
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 計算中心點 X 座標
            center_x = (x1 + x2) / 2
            cv2.circle(frame, (int(center_x), int((y1 + y2) / 2)), 5, (0, 0, 255), -1)

        # 4. 顯示測試畫面
        cv2.imshow("YOLOv8 SF6 Tracking Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_yolo_tracking()