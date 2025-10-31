import csv
import os
import shutil

path_val = '/root/autodl-tmp/RDED-main/dataset/imagenet/val_origin/'
img_list = os.listdir(path_val)

path_csv = '/root/autodl-tmp/RDED-main/dataset/imagenet/selected_imagenet.csv'
csvFile = open(path_csv, "r")
reader = csv.reader(csvFile)
path_root = '/root/autodl-tmp/RDED-main/dataset/imagenet/val/'

for item in reader:
        item_path = os.path.join(path_root,item[1])
        if reader.line_num == 1:
                continue
        if not os.path.exists(item_path):
                os.makedirs(item_path)
        path_img = os.path.join(path_val,item[2])
        shutil.copy(path_img, item_path)
