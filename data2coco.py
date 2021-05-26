
import re
import os
import cv2
import json
import itertools
import numpy as np
from glob import glob
import scipy.io as sio
from pycocotools import mask as cocomask
from PIL import Image
from os import listdir
def get_minVal(a,b,c,d):
    v1 = min(a,b)
    v2 = min(c,d)
    minval = min(v1,v2)
    return minval
def get_maxVal(a,b,c,d):
    v1 = max(a,b)
    v2 = max(c,d)
    maxval = max(v1,v2)
    return maxval

categories = [
    {
        "supercategory": "none",
        "name": "CH_str",
        "id": 0
    },
    {
        "supercategory": "none",
        "name": "CH_char",
        "id": 1
    },
    {
        "supercategory": "none",
        "name": "Eng_Digit_str",
        "id": 2
    },
    {
        "supercategory": "none",
        "name": "CH_Eng_Digit_str",
        "id": 3
    },
    {
        "supercategory": "none",
        "name": "CH_word",
        "id": 4
    },
    {
        "supercategory": "none",
        "name": "Oth",
        "id": 5
    },
    {
        "supercategory": "none",
        "name": "Not_Care",
        "id": 6
    },

]

phases = ["train", "valid"]
for phase in phases:
    image_id = 0
    annot_count = 0
    json_file = "my-dataset/annotations/{}.json".format(phase)

    res_file = {
        "categories": categories,
        "images": [],
        "annotations": []
    }

    images_path = f"./my-dataset/{phase}/"
    files = sorted(os.listdir(images_path))
    for f in files:
        img_path = os.path.join(images_path,f)
        img = Image.open(img_path)
        img_w, img_h = img.size
        img_elem = {"file_name": f,
                    "height": img_h,
                    "width": img_w,
                    "id": image_id}
        
        res_file["images"].append(img_elem)
        with open(f"my-dataset/gt/{f.split('.')[0]}.json",'r') as fr:
            data = json.load(fr)
            for item in data["shapes"]:
                points = []  # xl_up ,yl_up ,xr_up ,yr_up ,xr_dn ,yr_dn ,xl_dn ,yl_dn
                for point in item["points"]:
                    x,y = point
                    points.append(x)
                    points.append(y)
                xmin = int(get_minVal(points[0],points[2],points[4],points[6]))
                ymin = int(get_minVal(points[1],points[3],points[5],points[7]))
                xmax = int(get_maxVal(points[0],points[2],points[4],points[6]))
                ymax = int(get_maxVal(points[1],points[3],points[5],points[7]))
                w = xmax - xmin
                h = ymax - ymin
                area = w * h
                poly = [[xmin, ymin],
                        [xmax, ymin],
                        [xmax, ymax],
                        [xmin, ymax]]
                g_id = item["group_id"]
                if g_id ==255:
                    g_id = 6

                annot_elem = {
                        "id": annot_count,
                        "bbox": [
                            float(xmin),
                            float(ymin),
                            float(w),
                            float(h)
                        ],
                        "segmentation": list([poly]),
                        "image_id": image_id,
                        "ignore": 0,
                        "category_id": g_id,
                        "iscrowd": 0,
                        "area": float(area)
                }

                res_file["annotations"].append(annot_elem)
                annot_count += 1
                
        image_id += 1

    with open(json_file, "w") as f:
        json_str = json.dumps(res_file)
        f.write(json_str)

    # print("Processed {} {} images...".format(processed, phase))
print("Done.")
