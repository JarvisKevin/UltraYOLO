import os
from ultralytics import YOLO

def main(args):

    if args.data_txt_path is not None:
        with open(args.data_txt_path, 'r') as f:
            data = f.readlines()
        sources = [os.path.join(args.data_path, i.strip("\n")) for i in data]
    else:
        sources = args.data_path

    # Load a model
    model = YOLO(args.checkpoint)

    results = model.predict(source=sources,
                    batch=1,
                    # visualize=True,
                    imgsz=args.input_shape,
                    project=args.workdir,
                    save=True,
                    show_labels=False,
                    name=f"0712/{args.model_name}/{args.output_name}")

    # if isinstance(sources, list):
    #     for source in sources:
    #         # Train the model with 2 GPUs
    #         results = model.predict(source=source,
    #                                 batch=args.bs,
    #                                 # visualize=True,
    #                                 imgsz=args.input_shape,
    #                                 project=args.workdir,
    #                                 save=True,
    #                                 show_labels=False,
    #                                 name=f"0712/{args.model_name}/{args.output_name}")
    # else:
    #     results = model.predict(source=sources,
    #                     batch=args.bs,
    #                     # visualize=True,
    #                     imgsz=args.input_shape,
    #                     project=args.workdir,
    #                     save=True,
    #                     show_labels=False,
    #                     name=f"0712/{args.model_name}/{args.output_name}")


if __name__ == "__main__":

    import argparse

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='模型参数')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--data_txt_path', type=str, default=None)
    parser.add_argument('--input_shape', type=int, default=1824)
    parser.add_argument('--model_name', type=str, default="yolo11x-seg")
    parser.add_argument('--workdir', type=str, default="./line_seg")
    parser.add_argument('--output_name', type=str, default="./output_name")
    parser.add_argument('--device', nargs="+", type=str, default=["0"])
    parser.add_argument('--classes', nargs="+", type=str, default=None)
    parser.add_argument('--bs', type=int, default=24)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    main(args)

"""

/home/wenjunlin/workspace/UltraYOLO/line_det/0712/yolo11n/ori/y11n_3500_bs24_e100_lr3_0.99407,0.89002/weights/best.pt


python tools/classify_predict.py \
    --checkpoint /home/wenjunlin/workspace/UltraYOLO/line_det/0712/yolo11n/ori/y11n_3500_bs24_e100_lr3_0.99407,0.89002/weights/best.pt \
    --data_path /home/wenjunlin/workspace/UltraYOLO/222.png \
    --input_shape 2500 \
    --model_name yolo11n \
    --bs 1 \
    --device 0 \
    --workdir ./line_det \
    --output_name ori/y11n_3500_bs24_e100_lr3/predict/2500/val \
    --num_workers 10

    


python tools/classify_predict.py \
    --checkpoint /home/wenjunlin/workspace/UltraYOLO/line_det/0712/yolo11n/ori/y11n_3500_bs24_e100_lr3_0.99407,0.89002/weights/best.pt \
    --data_path /home/wenjunlin/workspace/20250724-134548_2.png \
    --input_shape 3500 \
    --model_name yolo11n \
    --bs 1 \
    --device 0 \
    --workdir ./line_det \
    --output_name ori/y11n_3500_bs24_e100_lr3/predict/3500/val \
    --num_workers 10

python tools/classify_predict.py \
    --checkpoint /home/wenjunlin/workspace/UltraYOLO/line_det/0712/yolo11n/ori/y11n_1500_bs64_e50_lr3_mosaic032_0.98866,0.8374/weights/best.pt \
    --data_path /home/wenjunlin/workspace/20250724-134548_2.png \
    --input_shape 1500 \
    --model_name yolo11n \
    --bs 1 \
    --device 0 \
    --workdir /home/wenjunlin/workspace/UltraYOLO/line_det/0712/yolo11n/ori/y11n_1500_bs64_e50_lr3_mosaic032_0.98866,0.8374 \
    --output_name predict/1500/val \
    --num_workers 10


python tools/classify_predict.py \
    --checkpoint /home/wenjunlin/workspace/UltraYOLO/line_det/0712/home/wenjunlin/workspace/UltraYOLO/ultralytics/cfg/models/11/yolo11_AddP2/yolo11/yolo11_AddP2/y11n_3500_bs24_e100_lr34/weights/best.pt \
    --data_path /home/wenjunlin/workspace/20250724-134548_2.png \
    --input_shape 3500 \
    --model_name yolo11n \
    --bs 1 \
    --device 0 \
    --workdir ./line_det \
    --output_name ori/y11n_3500_bs24_e100_lr3/predict/3500/val \
    --num_workers 10


python tools/classify_predict.py \
    --checkpoint /home/wenjunlin/workspace/UltraYOLO/line_det/0712/home/wenjunlin/workspace/UltraYOLO/ultralytics/cfg/models/11/yolo11_AddP2/yolo11/yolo11_AddP2/y11n_3500_bs24_e100_lr34/weights/best.pt \
    --data_path /home/wenjunlin/workspace/20250724-134548_2.png \
    --input_shape 1500 \
    --model_name yolo11n \
    --bs 1 \
    --device 0 \
    --workdir ./line_det \
    --output_name ori/y11n_3500_bs24_e100_lr3/predict/1500/val \
    --num_workers 10



python tools/classify_predict.py \
    --checkpoint /home/wenjunlin/workspace/UltraYOLO/line_seg/0712/yolo11s-seg/depth025width05/y11s_3500_bs24_e100_lr3/weights/best.pt \
    --data_path /home/wenjunlin/workspace/20250723-103949.jpg \
    --input_shape 3500 \
    --model_name yolo11s-seg \
    --bs 10 \
    --device 0 \
    --workdir ./line_seg \
    --output_name y11s_3500_bs24_e100_lr3/predict/3500/val \
    --num_workers 10


python tools/classify_predict.py \
    --checkpoint /home/wjl/workspace/ultralytics/line_seg/0712/yolo11x-seg/y11x_1824_bs32_e100_lr4/weights/best.pt \
    --data_path /home/wjl/workspace/segmentation/mmsegmentation-0.27.0/custom_tools/instance_annotations/yolo/ \
    --data_txt_path /home/wjl/workspace/segmentation/mmsegmentation-0.27.0/custom_tools/instance_annotations/yolo/val.txt \
    --input_shape 1824 \
    --model_name yolo11x-seg \
    --bs 3 \
    --device 0 \
    --workdir ./line_seg \
    --output_name y11x_1824_bs32_e100_lr4/predict/1824/val \
    --num_workers 10

python tools/classify_predict.py \
    --checkpoint /home/wjl/workspace/ultralytics/line_seg/0712/yolo11x-seg/y11x_1824_bs32_e100_lr4/weights/best.pt \
    --data_path /home/wjl/workspace/segmentation/mmsegmentation-0.27.0/custom_tools/instance_annotations/yolo/ \
    --data_txt_path /home/wjl/workspace/segmentation/mmsegmentation-0.27.0/custom_tools/instance_annotations/yolo/val.txt \
    --input_shape 1824 \
    --model_name yolo11x-seg \
    --bs 3 \
    --device 0 \
    --workdir ./line_seg \
    --output_name y11x_1824_bs32_e100_lr4/predict/1824/val \
    --num_workers 10

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