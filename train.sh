# 目标检测
# /home/wenjunlin/workspace/UltraYOLO/ultralytics/cfg/models/11/yolo11.yaml

# 大目标没法检测出来，说明感受野不够
# 1. 降低分辨率推理 2500 1500
	# 1500可行，但会多框，那1500对于小目标来说够用吗？可以实验一下
# 2. 加深网络提升感受野
	# 
# albu先去掉

# python tools/classify_train.py \
# 	--input_shape 2500 \
# 	--data_path /home/wenjunlin/workspace/UltraYOLO/0line/cfg/line.yaml \
# 	--model_name yolo11n \
# 	--bs 32 \
# 	--lr0 0.001 \
# 	--mosaic 0.3 \
# 	--device 0 1 2 3 4 5 6 7 \
# 	--epochs 50 \
# 	--workdir ./line_det \
# 	--output_name ori/y11n_2500_bs64_e50_lr3_mosaic03 \
# 	--num_workers 6

# python tools/classify_train.py \
# 	--input_shape 1500 \
# 	--data_path /home/wenjunlin/workspace/UltraYOLO/0line/cfg/line.yaml \
# 	--model_name yolo11n \
# 	--bs 32 \
# 	--lr0 0.001 \
# 	--mosaic 0.3 \
# 	--device 0 1 2 3 4 5 6 7 \
# 	--epochs 50 \
# 	--workdir ./line_det \
# 	--output_name ori/y11n_1500_bs64_e50_lr3_mosaic03 \
# 	--num_workers 6


# python tools/classify_train.py \
# 	--input_shape 3500 \
# 	--data_path ./0line/cfg/line.yaml \
# 	--model_name yolo11 \
# 	--bs 12 \
# 	--lr0 0.001 \
# 	--device 0 1 2 3 \
# 	--epochs 20 \
# 	--workdir ./line_det \
# 	--output_name reg_max/25/y11n_3500_bs24_e100_lr3 \
# 	--num_workers 8

# python tools/classify_train.py \
# 	--input_shape 3500 \
# 	--data_path /home/wenjunlin/workspace/UltraYOLO/0line/cfg/line.yaml \
# 	--model_name /home/wenjunlin/workspace/UltraYOLO/ultralytics/cfg/models/11/reg_max/25/yolo11 \
# 	--bs 32 \
# 	--lr0 0.001 \
# 	--device 0 1 2 3 4 5 6 7 \
# 	--epochs 100 \
# 	--workdir ./line_det \
# 	--output_name reg_max/25/y11n_3500_bs24_e100_lr3 \
# 	--num_workers 8

# python tools/classify_train.py \
# 	--input_shape 3500 \
# 	--data_path /home/wenjunlin/workspace/UltraYOLO/0line/cfg/line.yaml \
# 	--model_name /home/wenjunlin/workspace/UltraYOLO/ultralytics/cfg/models/11/reg_max/35/yolo11 \
# 	--bs 32 \
# 	--lr0 0.001 \
# 	--device 0 1 2 3 4 5 6 7 \
# 	--epochs 100 \
# 	--workdir ./line_det \
# 	--output_name reg_max/35/y11n_3500_bs24_e100_lr3 \
# 	--num_workers 8

# python tools/classify_train.py \
# 	--input_shape 3500 \
# 	--data_path /home/wenjunlin/workspace/UltraYOLO/0line/cfg/line.yaml \
# 	--model_name /home/wenjunlin/workspace/UltraYOLO/ultralytics/cfg/models/11/reg_max/45/yolo11 \
# 	--bs 32 \
# 	--lr0 0.001 \
# 	--device 0 1 2 3 4 5 6 7 \
# 	--epochs 100 \
# 	--workdir ./line_det \
# 	--output_name reg_max/45/y11n_3500_bs24_e100_lr3 \
# 	--num_workers 8

# python tools/classify_train.py \
# 	--input_shape 2500 \
# 	--data_path /home/wenjunlin/workspace/UltraYOLO/0line/cfg/line.yaml \
# 	--model_name /home/wenjunlin/workspace/UltraYOLO/ultralytics/cfg/models/11/reg_max/45/yolo11 \
# 	--bs 32 \
# 	--lr0 0.001 \
# 	--device 0 1 2 3 4 5 6 7 \
# 	--epochs 100 \
# 	--workdir ./line_det \
# 	--output_name reg_max/45/y11n_2500_bs24_e100_lr3 \
# 	--num_workers 8


# python tools/classify_train.py \
# 	--input_shape 2500 \
# 	--data_path /home/wenjunlin/workspace/UltraYOLO/0line/cfg/line.yaml \
# 	--model_name rtdetr-l \
# 	--bs 48 \
# 	--lr0 0.001 \
# 	--device 0 1 2 3 4 5 6 7 \
# 	--epochs 100 \
# 	--workdir ./line_det \
# 	--output_name rtdetr_l/baseline/y11n_2500_bs24_e100_lr3 \
# 	--num_workers 8


# python tools/classify_train.py \
# 	--input_shape 3500 \
# 	--data_path /home/wenjunlin/workspace/UltraYOLO/0line/cfg/line.yaml \
# 	--model_name yolo11 \
# 	--bs 64 \
# 	--lr0 0.001 \
# 	--device 0 1 2 3 4 5 6 7 \
# 	--epochs 100 \
# 	--workdir ./line_det \
# 	--output_name y11n_3500_bs64_e100_lr3_compress \
# 	--num_workers 8

## reg_max 25
# multi_scale 2200

# 不行再加P2
# python tools/classify_train.py \
# 	--input_shape 2500 \
# 	--data_path /home/wenjunlin/workspace/UltraYOLO/0line/cfg/line.yaml \
# 	--model_name yolo11m \
# 	--bs 49 \
# 	--lr0 0.0001 \
# 	--device 1 2 3 4 5 6 7 \
# 	--epochs 30 \
# 	--workdir ./line_det0908 \
# 	--output_name y11m_2500_bs49_e30_lr4_compress \
# 	--num_workers 6

# python tools/classify_train.py \
# 	--input_shape 2500 \
# 	--data_path /home/wenjunlin/workspace/UltraYOLO/0line/cfg/line.yaml \
# 	--model_name yolo11l \
# 	--bs 49 \
# 	--lr0 0.0001 \
# 	--device 1 2 3 4 5 6 7 \
# 	--epochs 30 \
# 	--workdir ./line_det0908 \
# 	--output_name y11l_2500_bs49_e30_lr4_compress \
# 	--num_workers 6

python tools/classify_train.py \
	--input_shape 2500 \
	--data_path /home/wenjunlin/workspace/UltraYOLO/0line/cfg/line.yaml \
	--model_name yolo11l \
	--bs 14 \
	--lr0 0.0001 \
	--device 1 2 3 4 5 6 7 \
	--epochs 50 \
	--workdir ./line_det0908 \
	--multi_scale True \
	--output_name y11l_2500_bs14_e50_lr4_compress_multiscale \
	--num_workers 6

python tools/classify_train.py \
	--input_shape 2500 \
	--data_path /home/wenjunlin/workspace/UltraYOLO/0line/cfg/line.yaml \
	--model_name yolo11l \
	--bs 14 \
	--lr0 0.0001 \
	--device 1 2 3 4 5 6 7 \
	--epochs 80 \
	--workdir ./line_det0908 \
	--multi_scale True \
	--output_name y11l_2500_bs14_e80_lr4_compress_multiscale \
	--num_workers 6

# python tools/classify_train.py \
# 	--input_shape 3500 \
# 	--data_path /home/wenjunlin/workspace/UltraYOLO/0line/cfg/line.yaml \
# 	--model_name yolo11m \
# 	--bs 21 \
# 	--lr0 0.0001 \
# 	--device 1 2 3 4 5 6 7 \
# 	--epochs 30 \
# 	--workdir ./line_det0905 \
# 	--output_name y11m_3500_bs21_e30_lr4_compress_regmax33 \
# 	--num_workers 6



# python tools/classify_train.py \
# 	--input_shape 3500 \
# 	--data_path /home/wenjunlin/workspace/UltraYOLO/0line/cfg/line.yaml \
# 	--model_name yolo11l \
# 	--bs 14 \
# 	--lr0 0.0001 \
# 	--device 1 2 3 4 5 6 7 \
# 	--epochs 30 \
# 	--workdir ./line_det0905 \
# 	--output_name y11l_3500_bs14_e30_lr4_compress_regmax50 \
# 	--num_workers 6

 
# multi scale TTA

# close_mosaic=10,
# copy_paste=0.5, 
# cutmix=0.0,
# degrees=10.0, 
# dropout=0.0
# fliplr=0.5, 
# flipud=0.5,
# mixup=0.5,
# mosaic=1.0,
# perspective=0.001

# freeze and then multiscale
# 