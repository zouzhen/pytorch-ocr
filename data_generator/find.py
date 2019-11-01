import os.path
import cv2
import pandas as pd

label_path = "/home/lzc274500/WorkSpace/ZOUZHEN/Pytorch/crnn_chinese_characters_rec/data/val.txt"
print('============================')
data = pd.read_csv(label_path,sep=' ',header=None,error_bad_lines=False)
print('============================')

find_data = data[data[0]=='20474828_1647735953.jpg'].values

print(find_data[0][1])
print(type(find_data[0][1]))