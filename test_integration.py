import cv2
import time
from sf6_env import SF6Env


def run_integration_test():
    """
    這是一個純測試腳本，用來確保：
    1. 畫面能被正確擷取並轉換。
    2. 按鍵能被正確送出而不會當機。
    3. AI 視角能正確涵蓋遊戲重要資訊 (血條)。
    """
    print("🚀 開始系統整合測試...")
    print("⚠️ 請在倒數結束前，將《快打旋風 6》切換為作用中視窗 (前景)！")

    for i in range(5, 0, -1):
        print(f"倒數 {i} 秒...")
        time.sleep(1)

    try:
        # 1. 初始化我們辛苦建立的遊戲環境
        env = SF6Env()
        obs, info = env.reset()

        print("✅ 環境初始化成功！開始隨機動作測試 (按下小視窗上的 'q' 鍵可提早結束)")

        # 2. 測試執行 100 步
        for step in range(100):
            # 讓 Gym 環境隨機挑選一個合法的按鍵動作
            action = env.action_space.sample()

            # 執行步驟 (這會觸發截圖、按鍵、算分數的完整迴圈)
            obs, reward, terminated, truncated, info = env.step(action)

            # 3. 將 AI 看到的畫面顯示出來 (視覺化 Debug)
            # obs 的形狀是 (1, 144, 256)，我們需要取第 0 個維度變成 (144, 256) 才能顯示
            display_img = obs[0]

            # 放大一點點比較好觀察 (放大 2 倍)
            display_img_enlarged = cv2.resize(display_img,  (512, 288))
            cv2.imshow("AI Vision Debug (Press 'q' to quit)", display_img_enlarged)

            # 暫停一下讓 OpenCV 更新視窗，同時捕捉你是否按下了 'q' 鍵
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("🛑 收到中斷指令，提前結束測試。")
                break

            if terminated or truncated:
                print("🔄 觸發回合結束條件，重置環境...")
                obs, info = env.reset()

        print("🎉 整合測試順利完成！沒有發生任何報錯。")

    except Exception as e:
        print(f"❌ 測試過程中發生錯誤：{e}")
        print("請檢查所有檔案是否都在同一個資料夾，以及 JSON 設定檔是否正確。")
    finally:
        # 清除 OpenCV 視窗
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_integration_test()