import os
import time
from stable_baselines3 import PPO
from sf6_env import SF6Env


def play_game():
    print("========================================")
    print("🎮 《快打旋風 6》AI 實戰展示模式啟動 🎮")
    print("========================================")

    # 1. 指定你要載入的模型檔案路徑
    # 這裡我們載入你剛剛訓練出來的 final_model，或者 checkpoint 檔案
    # 注意：副檔名 .zip 不一定要寫出來，SB3 會自動找
    model_path = "models/sf6_emergency_save"

    if not os.path.exists(f"{model_path}.zip"):
        print(f"❌ 找不到模型檔案: {model_path}.zip")
        print("請確認你已經跑完訓練，或者修改 model_path 換成 models 資料夾裡有的檔案。")
        return

    # 2. 初始化遊戲環境
    print("正在初始化遊戲環境...")
    env = SF6Env()

    # 3. 載入訓練好的大腦
    print(f"🧠 正在載入 AI 大腦: {model_path}...")
    model = PPO.load(model_path, env=env, device="cpu")

    # 4. 開始無限遊玩迴圈
    obs, info = env.reset()
    print("✅ 載入成功！AI 開始實戰 (按下 Ctrl+C 結束)")

    try:
        while True:
            # ⭐️ 關鍵：使用 predict 而不是 learn
            # deterministic=True 代表讓 AI 選擇「最確定/最佳」的動作，不再隨機瞎猜
            action, _states = model.predict(obs, deterministic=True)

            # 執行動作並觀察結果
            obs, reward, terminated, truncated, info = env.step(action)

            # 如果回合結束，環境會自動呼叫剛剛我們修改過的 reset() 去等待 7 秒
            if terminated or truncated:
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\n🛑 收到中斷指令，結束遊玩模式。")


if __name__ == "__main__":
    play_game()