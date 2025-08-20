import cv2
from ultralytics import YOLO

# 封面 乐谱分类
model_dir = '/home/wenjunlin/workspace/UltraYOLO/line_det/0712/yolo11n/ori/y11n_3500_bs24_e100_lr3_0.99407,0.89002/weights/best.pt'
model = YOLO(model_dir)
model.export(format="onnx", dynamic=True, half=True)

image_path = "/home/wenjunlin/workspace/UltraYOLO/22.png"
results = model(cv2.imread(image_path))
for result in results:
    print(result.probs.data)