# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 11:12:25 2020

@author: zhangjing
"""
import os
import torch
from PIL import ImageDraw
from PIL import Image
from collections import Counter
from cnn import MyModel,load_tensor,resize_image,image_to_tensor,calc_iou,merge_box,map_box_to_original_image

# 用于启用 GPU 支持
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 分类列表
CLASSES = [ "other", "with_mask", "without_mask", "mask_weared_incorrect" ]
# 创建模型实例
model = MyModel().to(device)
# 分析目标的图片所在的文件夹
IMAGE_DIR = "./images"

model.load_state_dict(load_tensor("model.pt"))

# 判断是否应该合并重叠区域的重叠率阈值
IOU_MERGE_THRESHOLD = 0.35

def eval_model():
    """使用训练好的模型"""
    # 创建模型实例，加载训练好的状态，然后切换到验证模式
    model = MyModel().to(device)
    model.load_state_dict(load_tensor("model.pt"))
    model.eval()

    # 询问图片路径，并显示所有可能是人脸的区域
    for filename in os.listdir(IMAGE_DIR):
        if not filename:
            continue
        # 构建输入
        with Image.open(IMAGE_DIR+'/'+filename) as img_original: # 加载原始图片
            sw, sh = img_original.size # 原始图片大小
            img = resize_image(img_original) # 缩放图片
            img_output = img_original.copy() # 复制图片，用于后面添加标记
            tensor_in = image_to_tensor(img)
        # 预测输出
        cls_result = model(tensor_in.unsqueeze(0).to(device))[-1][0]
        if cls_result==None:
            with open("./input/detection-results/"+filename.split(".")[0]+".txt", "w") as out:
                out.write("")
            continue
        # 合并重叠的结果区域, 结果是 [ [标签列表, 合并后的区域], ... ]
        final_result = []
        for label, box in cls_result:
            for index in range(len(final_result)):
                exists_labels, exists_box = final_result[index]
                if calc_iou(box, exists_box) > IOU_MERGE_THRESHOLD:
                    exists_labels.append(label)
                    final_result[index] = (exists_labels, merge_box(box, exists_box))
                    break
            else:
                final_result.append(([label], box))
        
        # 合并标签 (重叠区域的标签中数量最多的分类为最终分类)
        for index in range(len(final_result)):
            labels, box = final_result[index]
            final_label = Counter(labels).most_common(1)[0][0]
            final_result[index] = (final_label, box)
        # 标记在图片上
        draw = ImageDraw.Draw(img_output)
        txt_output=""
        for label, box in final_result:
            x, y, w, h = map_box_to_original_image(box, sw, sh)
            draw.rectangle((x, y, x+w, y+h), outline="#FF0000")
            draw.text((x, y-10), CLASSES[label], fill="#FF0000")
            txt_output+=CLASSES[label]
            txt_output+=" "
            txt_output+=str(calc_iou(box, exists_box))
            txt_output+=" "
            txt_output+=str(x)
            txt_output+=" "
            txt_output+=str(y)
            txt_output+=" "
            txt_output+=str(x+w)
            txt_output+=" "
            txt_output+=str(y+h)
            txt_output+="\n"
        with open("./input/detection-results/"+filename.split(".")[0]+".txt", "w") as out:
            out.write(txt_output)
            

if __name__ == "__main__":
    eval_model()