import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack  # 🌟 匯入所需模組
from sf6_env import SF6Env


def main():
    print("========================================")
    print("🤖 《快打旋風 6》AI 智慧訓練程式啟動 (Frame Stacking 版本) 🤖")
    print("========================================")

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    print("正在初始化遊戲環境與視覺擷取模組...")

    # 🌟 步驟 1: 初始化基本環境
    def make_env():
        return SF6Env()

    # 🌟 步驟 2: 使用 DummyVecEnv 包裝環境，並套用 VecFrameStack
    # n_stack=4 代表將過去 4 幀畫面疊加，讓 AI 擁有短暫記憶
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=models_dir,
        name_prefix="sf6_stacked_model",  # 更改前綴以區分舊模型
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # ==========================================
    # ⭐️ 由於改變了觀察空間大小 (1 通道 -> 4 通道)，
    # 我們必須建立一個全新的大腦，無法載入舊模型。
    # ==========================================
    model_name = "sf6_stacked_emergency_save"
    load_model_path = os.path.join(models_dir, model_name)

    if os.path.exists(f"{load_model_path}.zip"):
        print(f"🧠 找到過去的 Frame Stack 訓練紀錄！正在載入：{load_model_path}.zip ...")
        model = PPO.load(load_model_path, env=env, device="cpu")
    else:
        print("🌱 找不到過去的紀錄，建立一個全新的 PPO 神經網路大腦 (支援連續視覺)...")
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            device="cpu",
            learning_rate=0.0003,
            n_steps=4096,
            batch_size=64,
            tensorboard_log="./tensorboard_logs/"
        )
    # ==========================================

    total_timesteps = 1000000
    print(f"\n🚀 開始訓練！預計進行 {total_timesteps} 步。")
    print("⚠️ 提醒：請確保遊戲視窗在左上角 (1280x720)，並保持在最上層。")
    print("按下 Ctrl+C 可以隨時安全中斷訓練並存檔。\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=False
        )

        final_model_path = os.path.join(models_dir, "sf6_stacked_final_model")
        model.save(final_model_path)
        print(f"\n✅ 訓練完美結束！最新模型已儲存至 {final_model_path}.zip")

    except KeyboardInterrupt:
        print("\n🛑 收到手動中斷指令 (Ctrl+C)！")
        emergency_save_path = os.path.join(models_dir, "sf6_stacked_emergency_save")
        model.save(emergency_save_path)
        print(f"💾 已緊急儲存目前最新進度至 {emergency_save_path}.zip")


if __name__ == "__main__":
    main()