from ultralytics import YOLO

# 加载YOLOv8s模型
model = YOLO('yolov8s.pt')

# 配置训练参数
model.train(
    data='textile_defect.yaml',  # 数据集配置文件
    epochs=100,
    imgsz=640,  # 输入分辨率
    batch=16,   # 批次大小
    lr0=0.01,   # 初始学习率
    cos_lr=True,  # 余弦退火学习率调度器
    patience=10,  # 早停机制：10个epoch无提升停止
    device=0      # 使用GPU
)

# 验证模型
results = model.val()
print(f"mAP@0.5: {results.box.map50:.3f}")