import os
from ultralytics import YOLO, RTDETR

def main(args):
    data_path = args.data_path
    pretrained = f'/home/wenjunlin/workspace/UltraYOLO/pretrained/{args.model_name}.pt'

    # Load a model
    if "yolo" in args.model_name:
        model = YOLO(args.model_name+'.yaml')
    elif "rtdetr" in args.model_name:
        model = RTDETR(args.model_name+'.yaml')
    
    if os.path.exists(pretrained):
        model = model.load(pretrained)  # load a pretrained model (recommended for training)
    else:
        model = model.load("/home/wenjunlin/workspace/UltraYOLO/pretrained/yolo11n.pt")

    # Train the model with 2 GPUs
    results = model.train(data=data_path, 
                            epochs=args.epochs, 
                            imgsz=args.input_shape, 
                            multi_scale=args.multi_scale,
                            device=args.device, 
                            lr0=args.lr0,
                            batch=args.bs,
                            optimizer="Adam", 
                            cos_lr=True,
                            workers=args.num_workers,
                            augment=True,
                            project=args.workdir,
                            classes=args.classes,
                            name=f"0712/{args.model_name}/{args.output_name}",
                            close_mosaic=10,
                            copy_paste=0.5, 
                            cutmix=0.0,
                            degrees=10.0, 
                            dropout=0.0,
                            fliplr=0.5, 
                            flipud=0.5,
                            mixup=1, #0.5,
                            mosaic=args.mosaic, #1.0,
                            perspective=0.0
                            )

if __name__ == "__main__":

    import argparse

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='模型参数')
    parser.add_argument('--data_path', type=str, default="/home/wjl/workspace/ultralytics/0line/cfg/line.yaml")
    parser.add_argument('--input_shape', type=int, default=1824)
    parser.add_argument('--multi_scale', type=bool, default=False)
    parser.add_argument('--model_name', type=str, default="yolo11x-seg")
    parser.add_argument('--workdir', type=str, default="./line_seg")
    parser.add_argument('--output_name', type=str, default="./output_name")
    parser.add_argument('--device', nargs="+", type=str, default=["0", "1", "2", "3", "4", "5", "6", "7"])
    parser.add_argument('--classes', nargs="+", type=str, default=None)
    parser.add_argument('--lr0', type=float, default=0.001)
    parser.add_argument('--mosaic', type=float, default=1.0)
    parser.add_argument('--bs', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    main(args)

""" 

python tools/classify_train.py \
    --input_shape 1824 \
    --data_path /home/wjl/workspace/ultralytics/0line/cfg/line.yaml \
    --model_name yolo11x-seg \
    --bs 24 \
    --lr0 0.001 \
    --device 0 1 2 3 4 5 6 7 \
    --epochs 100 \
    --workdir ./line_seg \
    --output_name y11x_1824_bs32_e100_lr3 \
    --num_workers 8    

python tools/classify_train.py \
    --input_shape 1800 \
    --data_path /home/wjl/workspace/ultralytics/0line/cfg/line.yaml \
    --model_name yolo12l-seg \
    --bs 8 \
    --lr0 0.001 \
    --device 0 1 2 3 4 5 6 7 \
    --epochs 100 \
    --workdir ./line_seg \
    --output_name y12l_1824_bs8_e100_lr3 \
    --num_workers 8

"""