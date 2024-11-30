import torch
from pathlib import Path

# 原始权重文件路径
weights_path = "your_model.pt"

# 指定保存的目标路径（使用 Path 来处理路径）
save_dir = Path("E:/saved_weights")  # 替换为你的目标保存路径
save_dir.mkdir(parents=True, exist_ok=True)  # 确保路径存在，不存在时自动创建

# 保存的新权重文件名
save_path = save_dir / "new_model.pt"

# 加载模型
model = torch.load(weights_path, map_location="cpu")

# 保存到指定路径
torch.save(model, save_path)

print(f"Model successfully saved to {save_path}")
