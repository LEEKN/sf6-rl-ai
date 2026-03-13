import os
from stable_baselines3 import PPO

class ModelTrainer:
    def __init__(self, base_model_path="models/base_model"):
        self.base_model_path = base_model_path
        os.makedirs("models", exist_ok=True)

    def train_base_model(self, base_env, total_timesteps=1000000):
        print("🧠 開始訓練基礎模型...")
        model = PPO("CnnPolicy", base_env, verbose=1)
        model.learn(total_timesteps=total_timesteps)
        model.save(self.base_model_path)
        return model

    def train_matchup_specific_model(self, specific_env, opponent_name, total_timesteps=500000):
        print(f"🎯 開始針對 [{opponent_name}] 進行特訓...")
        try:
            model = PPO.load(self.base_model_path, env=specific_env)
        except FileNotFoundError:
            print("❌ 找不到基礎模型！")
            return
        model.learn(total_timesteps=total_timesteps)
        model.save(f"models/model_vs_{opponent_name}")
        return model