import os
import time
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from sf6_env import SF6Env
from action_manager import SF6ActionManagerAsync

# ==========================================
# 🌟 AI 訓練全域設定中心
# ==========================================
AI_SIDE = "P1"
MATCH_MODE = "versus"
AUTO_PAUSE = False
N_STEPS = 4096
TENSORBOARD_LOG_DIR = "./tensorboard_logs/"  # 🌟 新增 Log 資料夾設定


# ==========================================

# 🌟 TODO 7: 新增 TensorBoard 攔截器
class TensorboardCallback(BaseCallback):
    """
    自訂的 Callback，用來在回合結束時將戰鬥數據推送到 TensorBoard。
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # 檢查這一步是否導致回合結束 (done)
        if self.locals.get("dones")[0]:
            # 取得環境傳出的 info 字典
            info = self.locals.get("infos")[0]

            # 如果裡面有我們剛才寫的結算資料，就畫圖！
            if "episode_damage_dealt" in info:
                # 記錄到 match (對戰) 分類下
                self.logger.record("match/damage_dealt", info["episode_damage_dealt"])
                self.logger.record("match/damage_taken", info["episode_damage_taken"])
                self.logger.record("match/win_rate", info["win"])
                self.logger.record("match/round_length_steps", info["match_length"])
        return True


# ... (AutoPauseCallback 保持不變) ...

def main():
    print("========================================")
    print(f"🤖 SF6 AI 啟動 (多模態 + 戰情室版) | 陣營: {AI_SIDE} | 模式: {MATCH_MODE} 🤖")
    print("========================================")

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)  # 🌟 確保 Log 資料夾存在

    env = SF6Env(ai_side=AI_SIDE, match_mode=MATCH_MODE)

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=models_dir, name_prefix="sf6_multi_lstm_model")
    tb_callback = TensorboardCallback()  # 🌟 實例化我們的畫圖器

    # 🌟 把 tb_callback 加入名單
    if AUTO_PAUSE:
        callback_list = CallbackList([checkpoint_callback, tb_callback, AutoPauseCallback()])
    else:
        callback_list = CallbackList([checkpoint_callback, tb_callback])

    model_name = "sf6_multi_lstm_emergency_save"
    load_model_path = os.path.join(models_dir, model_name)

    if os.path.exists(f"{load_model_path}.zip"):
        print(f"🧠 載入多模態舊大腦：{load_model_path}.zip ...")
        model = RecurrentPPO.load(
            load_model_path, env=env, device="cpu",
            custom_objects={"ent_coef": 0.05},
            tensorboard_log=TENSORBOARD_LOG_DIR  # 🌟 告訴模型去哪裡寫 Log
        )
    else:
        print("🌱 建立全新的 MultiInputLstmPolicy 神經網路大腦...")
        model = RecurrentPPO(
            "MultiInputLstmPolicy", env, verbose=1, device="cpu",
            learning_rate=0.0003, n_steps=N_STEPS, batch_size=64,
            ent_coef=0.05,
            tensorboard_log=TENSORBOARD_LOG_DIR  # 🌟 告訴模型去哪裡寫 Log
        )

    total_timesteps = 1000000
    print(f"\n🚀 開始訓練！按下 Ctrl+C 可隨時安全中斷存檔。\n")

    try:
        model.learn(total_timesteps=total_timesteps, callback=callback_list, reset_num_timesteps=False)
        model.save(os.path.join(models_dir, "sf6_multi_lstm_final"))
    except KeyboardInterrupt:
        print("\n🛑 手動中斷！已緊急儲存。")
        model.save(os.path.join(models_dir, "sf6_multi_lstm_emergency_save"))


if __name__ == "__main__":
    main()