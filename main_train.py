import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from sf6_env import SF6Env
from action_manager import SF6ActionManagerAsync

# ==========================================
# 🌟 AI 訓練全域設定中心 (Config Center) 🌟
# ==========================================
AI_SIDE = "P1"  # 設定 AI 扮演哪一邊："P1" (左) 或 "P2" (右)
MATCH_MODE = "versus"  # "training" (訓練場，無FIGHT動畫) 或 "versus" (實戰，有FIGHT動畫)
AUTO_PAUSE = False  # 迭代時是否自動按暫停？ (對戰電腦設 True，對戰真人設 False)
N_STEPS = 8192  # 每次迭代的步數 (8192步約10分鐘，足以收集 1~2 個 BO3 的資料)


# ==========================================

class AutoPauseCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.action_manager = SF6ActionManagerAsync("sf6_moves.json")
        self.is_paused = False

    def on_rollout_end(self) -> None:
        print("\n⏸️ [系統攔截] 準備進行模型迭代！按下暫停鍵...")
        self.action_manager.keyboard.execute_sequence(['enter'])  # 替換成你的暫停鍵
        self.is_paused = True

    def on_rollout_start(self) -> None:
        if self.is_paused:
            print("▶️ [系統攔截] 模型迭代完畢！解除暫停，回到戰場...")
            self.action_manager.keyboard.execute_sequence(['enter'])
            time.sleep(0.5)
            self.is_paused = False

    def _on_step(self) -> bool: return True


def main():
    print("========================================")
    print(f"🤖 SF6 AI 啟動 | 陣營: {AI_SIDE} | 模式: {MATCH_MODE} 🤖")
    print("========================================")

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # 🌟 將設定傳入環境
    env = SF6Env(ai_side=AI_SIDE, match_mode=MATCH_MODE)

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=models_dir, name_prefix="sf6_base_model")

    # 根據設定決定是否掛載暫停攔截器
    if AUTO_PAUSE:
        callback_list = CallbackList([checkpoint_callback, AutoPauseCallback()])
    else:
        callback_list = CallbackList([checkpoint_callback])

    model_name = "sf6_emergency_save"
    load_model_path = os.path.join(models_dir, model_name)

    if os.path.exists(f"{load_model_path}.zip"):
        print(f"🧠 載入舊大腦：{load_model_path}.zip ...")
        model = PPO.load(load_model_path, env=env, device="cpu")
    else:
        print("🌱 建立全新的 PPO 神經網路大腦...")
        model = PPO(
            "CnnPolicy", env, verbose=1, device="cpu",
            learning_rate=0.0003, n_steps=N_STEPS, batch_size=64
        )

    total_timesteps = 1000000
    print(f"\n🚀 開始訓練！按下 Ctrl+C 可隨時安全中斷存檔。\n")

    try:
        model.learn(total_timesteps=total_timesteps, callback=callback_list, reset_num_timesteps=False)
        model.save(os.path.join(models_dir, "sf6_final_model"))
    except KeyboardInterrupt:
        print("\n🛑 手動中斷！已緊急儲存。")
        model.save(os.path.join(models_dir, "sf6_emergency_save"))


if __name__ == "__main__":
    main()