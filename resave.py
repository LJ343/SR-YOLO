import torch
from pathlib import Path

# 原始权重文件路径
weights_path = "/root/YOLOv11New/runs/detect/Driving_v11SwinTransformer/train/exp2/weights/last.pt"

# 指定保存的目标路径（使用 Path 来处理路径）
save_dir = Path("/root/YOLOv11New/runs/detect/Driving_v11SwinTransformer/train/exp2/weights")  # 替换为你的目标保存路径
save_dir.mkdir(parents=True, exist_ok=True)  # 确保路径存在，不存在时自动创建

# 保存的新权重文件名
save_path = save_dir / "new_last.pt"

# 加载模型
model = torch.load(weights_path, map_location="cpu")

# 保存到指定路径
torch.save(model, save_path)

print(f"Model successfully saved to {save_path}")
