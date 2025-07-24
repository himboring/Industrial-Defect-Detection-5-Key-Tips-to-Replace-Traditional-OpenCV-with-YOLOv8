import cv2
import numpy as np
from ultralytics import YOLO

# 加载ONNX模型
model = YOLO('yolov8s.onnx', task='detect')

# 后处理函数：结合NMS和OpenCV形态学操作
def post_process_defects(image, results, conf_thres=0.3, iou_thres=0.5):
    # 提取推理结果
    boxes = results[0].boxes.xyxy.numpy()  # 边界框坐标
    scores = results[0].boxes.conf.numpy()  # 置信度
    valid_indices = scores > conf_thres  # 过滤低置信度
    
    # 非极大值抑制（NMS）
    if len(boxes) > 0:
        valid_boxes = boxes[valid_indices]
        valid_scores = scores[valid_indices]
        keep = cv2.dnn.NMSBoxes(valid_boxes.tolist(), valid_scores.tolist(), conf_thres, iou_thres)
        valid_boxes = valid_boxes[[i[0] for i in keep]] if len(keep) > 0 else np.array([])

    # 生成掩码并应用形态学操作
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for box in valid_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    # 形态学闭运算，填补小空洞
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask, valid_boxes

# 示例：推理并后处理
image = cv2.imread('plastic_part.jpg')
results = model(image, conf=0.3, iou=0.5)
mask, detected_boxes = post_process_defects(image, results)
cv2.imwrite('defect_mask.jpg', mask)