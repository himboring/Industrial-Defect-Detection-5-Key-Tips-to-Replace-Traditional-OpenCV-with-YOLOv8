import albumentations as A
import cv2
import os

# 定义可复用的数据增强pipeline
transform = A.Compose([
    A.RandomCrop(width=640, height=640, p=0.5),  # 随机裁剪模拟局部视角
    A.Rotate(limit=45, p=0.6),  # 随机旋转模拟不同拍摄角度
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),  # 光照变化
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),  # 添加高斯噪声模拟传感器噪声
    A.Blur(blur_limit=5, p=0.3),  # 模糊处理模拟镜头失焦
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),  # 色调调整
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 示例：对单张图像和标注应用增强
def augment_image(image_path, label_path, output_dir, num_aug=5):
    image = cv2.imread(image_path)
    with open(label_path, 'r') as f:
        bboxes = [list(map(float, line.split()[1:])) for line in f.readlines()]
        class_labels = [int(line.split()[0]) for line in f.readlines()]
    
    for i in range(num_aug):
        # 应用增强
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_image = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_labels = augmented['class_labels']
        
        # 保存增强后的图像和标注
        base_name = os.path.basename(image_path).split('.')[0]
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_aug_{i}.jpg"), aug_image)
        with open(os.path.join(output_dir, f"{base_name}_aug_{i}.txt"), 'w') as f:
            for bbox, label in zip(aug_bboxes, aug_labels):
                f.write(f"{label} {' '.join(map(str, bbox))}\n")

# 示例：增强单张电路板图像
augment_image('circuit_board.jpg', 'circuit_board.txt', 'augmented_dataset/')