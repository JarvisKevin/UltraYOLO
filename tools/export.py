import cv2
from ultralytics import YOLO

# 封面 乐谱分类
# 202508 
# '/home/wenjunlin/workspace/UltraYOLO/line_det/0712/yolo11n/ori/y11n_3500_bs24_e100_lr3_0.99407,0.89002/weights/best.pt'

# 0908 
#  "/home/wenjunlin/workspace/UltraYOLO/line_det0905/yolo11l/y11l_3500_bs14_e30_lr4_compress_0.99414,0.88484/weights/best.pt"
#  2500 "/home/wenjunlin/workspace/UltraYOLO/line_det0908/yolo11l/y11l_2500_bs49_e30_lr4_compress_regmax25_0.9939,0.87193/weights/best.pt"

# 0909 
# /home/wenjunlin/workspace/UltraYOLO/line_det0908/yolo11l/y11l_2500_bs14_e50_lr4_compress_multiscale_0.99413,0.88506/weights/best.pt

model_dir = "/home/wenjunlin/workspace/UltraYOLO/line_det0908/yolo11l/y11l_2500_bs14_e50_lr4_compress_multiscale_0.99413,0.88506/weights/best.pt"
model = YOLO(model_dir)
model.export(format="onnx", dynamic=True, half=True, device="0")

image_path = "/home/wenjunlin/workspace/UltraYOLO/22.png"
results = model(cv2.imread(image_path))
for result in results:
    # print(result.probs.data)
    print(result.probs)


