import cv2
import numpy as np
import os

# 合成划痕图像
def generate_synthetic_scratch(image_path, output_path, num_synthetic=50):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    annotations = []

    for i in range(num_synthetic):
        # 随机生成划痕
        synthetic_img = image.copy()
        x1, y1 = np.random.randint(100, w-100), np.random.randint(100, h-100)
        x2, y2 = x1 + np.random.randint(50, 200), y1 + np.random.randint(-50, 50)
        cv2.line(synthetic_img, (x1, y1), (x2, y2), (255, 255, 255), thickness=np.random.randint(1, 5))
        
        # 生成YOLO格式标注
        x_center = (x1 + x2) / 2 / w
        y_center = (y1 + y2) / 2 / h
        width = abs(x2 - x1) / w
        height = abs(y2 - y1) / h
        annotations.append(f"0 {x_center} {y_center} {width} {height}")
        
        # 保存合成图像
        cv2.imwrite(os.path.join(output_path, f"synthetic_{i}.jpg"), synthetic_img)
        with open(os.path.join(output_path, f"synthetic_{i}.txt"), 'w') as f:
            f.write("\n".join(annotations))
        annotations.clear()

# 示例：生成合成数据
generate_synthetic_scratch('base_metal.jpg', 'synthetic_dataset/')