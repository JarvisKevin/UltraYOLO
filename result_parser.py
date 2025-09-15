import pandas as pd

res_dir = "/home/wenjunlin/workspace/UltraYOLO/line_seg/0712/yolo11m-seg/depth025width05/y11m_3500_bs16_e100_lr3/results.csv"

df = pd.read_csv(res_dir)
print(df.columns)

df = df[["epoch"] + [i for i in df.columns if "metrics" in i]]

df = df.sort_values(by=["metrics/mAP50-95(B)", "metrics/mAP50-95(M)"])
print(df.iloc[-1])