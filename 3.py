from ultralytics import YOLO

# 加载YOLOv8n模型
model = YOLO('yolov8n.pt')

# 配置训练参数
model.train(
    data='aluminum_defect.yaml',  # 数据集配置文件
    epochs=50,
    imgsz=416,  # 输入分辨率
    batch=32,   # 批次大小适配Raspberry Pi内存
    lr0=0.01,  # 初始学习率
    cos_lr=True,  # 余弦退火学习率
    device='cpu'  # Raspberry Pi使用CPU
)

# 保存微调后的模型
model.save('yolov8n_aluminum.pt')