# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 11:29:53 2020

@author: scum
"""
import os
import sys
import torch
import gzip
import itertools
import random
import numpy
import math
import pandas
import json
from PIL import Image
from PIL import ImageDraw
from torch import nn
from matplotlib import pyplot
from collections import defaultdict
import xml.etree.cElementTree as ET
from collections import Counter
from sys import exit

# 缩放图片的大小
IMAGE_SIZE = (256, 192)
# 分析目标的图片所在的文件夹
IMAGE_DIR = "./archive/images"
# 定义各个图片中人脸区域与分类的 CSV 文件
ANNOTATION_DIR = "./archive/annotations"
# 分类列表
CLASSES = [ "other", "with_mask", "without_mask", "mask_weared_incorrect" ]
CLASSES_MAPPING = { c: index for index, c in enumerate(CLASSES) }
# 判断是否存在对象使用的区域重叠率的阈值
IOU_POSITIVE_THRESHOLD = 0.35
IOU_NEGATIVE_THRESHOLD = 0.10
# 判断是否应该合并重叠区域的重叠率阈值
IOU_MERGE_THRESHOLD = 0.35

# 用于启用 GPU 支持
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicBlock(nn.Module):
    """ResNet 使用的基础块"""
    expansion = 1 # 定义这个块的实际出通道是 channels_out 的几倍，这里的实现固定是一倍
    def __init__(self, channels_in, channels_out, stride):
        super().__init__()
        # 生成 3x3 的卷积层
        # 处理间隔 stride = 1 时，输出的长宽会等于输入的长宽，例如 (32-3+2)//1+1 == 32
        # 处理间隔 stride = 2 时，输出的长宽会等于输入的长宽的一半，例如 (32-3+2)//2+1 == 16
        # 此外 resnet 的 3x3 卷积层不使用偏移值 bias
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(channels_out))
        # 再定义一个让输出和输入维度相同的 3x3 卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels_out))
        # 让原始输入和输出相加的时候，需要维度一致，如果维度不一致则需要整合
        self.identity = nn.Sequential()
        if stride != 1 or channels_in != channels_out * self.expansion:
            self.identity = nn.Sequential(
                nn.Conv2d(channels_in, channels_out * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels_out * self.expansion))

    def forward(self, x):
        # x => conv1 => relu => conv2 => + => relu
        # |                              ^
        # |==============================|
        tmp = self.conv1(x)
        tmp = nn.functional.relu(tmp, inplace=True)
        tmp = self.conv2(tmp)
        tmp += self.identity(x)
        y = nn.functional.relu(tmp, inplace=True)
        return y

class MyModel(nn.Module):
    """Faster-RCNN (基于 ResNet-18 的变种)"""
    Anchors = None # 锚点列表，包含 锚点数量 * 形状数量 的范围
    AnchorSpan = 8 # 锚点之间的距离，应该等于原有长宽 / resnet 输出长宽
    AnchorScales = (0.5, 1, 2, 3, 4, 5, 6) # 锚点对应区域的缩放比例列表
    AnchorAspects = ((1, 2), (1, 1), (2, 1)) # 锚点对应区域的长宽比例列表
    AnchorBoxes = len(AnchorScales) * len(AnchorAspects) # 每个锚点对应的形状数量

    def __init__(self):
        super().__init__()
        # 抽取图片各个区域特征的 ResNet (除去 AvgPool 和全连接层)
        # 和 Fast-RCNN 例子不同的是输出的长宽会是原有的 1/8，后面会根据锚点与 affine_grid 截取区域
        # 此外，为了可以让模型跑在 4GB 显存上，这里减少了模型的通道数量
        # 注意:
        # RPN 使用的模型和标签分类使用的模型需要分开，否则会出现无法学习 (RPN 总是输出负) 的问题
        self.previous_channels_out = 4
        self.rpn_resnet = nn.Sequential(
            nn.Conv2d(3, self.previous_channels_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.previous_channels_out),
            nn.ReLU(inplace=True),
            self._make_layer(BasicBlock, channels_out=16, num_blocks=2, stride=1),
            self._make_layer(BasicBlock, channels_out=32, num_blocks=2, stride=2),
            self._make_layer(BasicBlock, channels_out=64, num_blocks=2, stride=2),
            self._make_layer(BasicBlock, channels_out=128, num_blocks=2, stride=2))
        self.previous_channels_out = 4
        self.cls_resnet = nn.Sequential(
            nn.Conv2d(3, self.previous_channels_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.previous_channels_out),
            nn.ReLU(inplace=True),
            self._make_layer(BasicBlock, channels_out=16, num_blocks=2, stride=1),
            self._make_layer(BasicBlock, channels_out=32, num_blocks=2, stride=2),
            self._make_layer(BasicBlock, channels_out=64, num_blocks=2, stride=2),
            self._make_layer(BasicBlock, channels_out=128, num_blocks=2, stride=2))
        self.features_channels = 128
        # 根据区域特征生成各个锚点对应的对象可能性的模型
        self.rpn_labels_model = nn.Sequential(
            nn.Linear(self.features_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, MyModel.AnchorBoxes*2))
        # 根据区域特征生成各个锚点对应的区域偏移的模型
        self.rpn_offsets_model = nn.Sequential(
            nn.Linear(self.features_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, MyModel.AnchorBoxes*4))
        # 选取可能出现对象的区域需要的最小可能性
        self.rpn_score_threshold = 0.9
        # 每张图片最多选取的区域列表
        self.rpn_max_candidates = 32
        # 根据区域截取特征后缩放到的大小
        self.pooling_size = 7
        # 根据区域特征判断分类的模型
        self.cls_labels_model = nn.Sequential(
            nn.Linear(self.features_channels * (self.pooling_size ** 2), 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, len(CLASSES)))
        # 根据区域特征再次生成区域偏移的模型，注意区域偏移会针对各个分类分别生成
        self.cls_offsets_model = nn.Sequential(
            nn.Linear(self.features_channels * (self.pooling_size ** 2), 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, len(CLASSES)*4))

    def _make_layer(self, block_type, channels_out, num_blocks, stride):
        """创建 resnet 使用的层"""
        blocks = []
        # 添加第一个块
        blocks.append(block_type(self.previous_channels_out, channels_out, stride))
        self.previous_channels_out = channels_out * block_type.expansion
        # 添加剩余的块，剩余的块固定处理间隔为 1，不会改变长宽
        for _ in range(num_blocks-1):
            blocks.append(block_type(self.previous_channels_out, self.previous_channels_out, 1))
            self.previous_channels_out *= block_type.expansion
        return nn.Sequential(*blocks)

    @staticmethod
    def _generate_anchors(span):
        """根据锚点和形状生成锚点范围列表"""
        w, h = IMAGE_SIZE
        anchors = []
        for x in range(0, w, span):
            for y in range(0, h, span):
                xcenter, ycenter = x + span / 2, y + span / 2
                for scale in MyModel.AnchorScales:
                    for ratio in MyModel.AnchorAspects:
                        ww = span * scale * ratio[0]
                        hh = span * scale * ratio[1]
                        xx = xcenter - ww / 2
                        yy = ycenter - hh / 2
                        xx = max(int(xx), 0)
                        yy = max(int(yy), 0)
                        ww = min(int(ww), w - xx)
                        hh = min(int(hh), h - yy)
                        anchors.append((xx, yy, ww, hh))
        return anchors

    @staticmethod
    def _roi_crop(features, rois, pooling_size):
        """根据区域截取特征，每次只能处理单张图片"""
        width, height = IMAGE_SIZE
        theta = []
        results = []
        for roi in rois:
            x1, y1, w, h = roi
            x2, y2 = x1 + w, y1 + h
            theta = [[
                [
                    (y2 - y1) / height,
                    0,
                    (y2 + y1) / height - 1
                ],
                [
                    0,
                    (x2 - x1) / width,
                    (x2 + x1) / width - 1
                ]
            ]]
            theta_tensor = torch.tensor(theta)
            grid = nn.functional.affine_grid(
                theta_tensor,
                torch.Size((1, 1, pooling_size, pooling_size)),
                align_corners=False).to(device)
            result = nn.functional.grid_sample(
                features.unsqueeze(0), grid, align_corners=False)
            results.append(result)
        if not results:
            return None
        results = torch.cat(results, dim=0)
        return results

    def forward(self, x):
        # ***** 抽取特征部分 *****
        # 分别抽取 RPN 和标签分类使用的特征
        # 维度是 B,128,W/8,H/8
        rpn_features_original = self.rpn_resnet(x)
        # 维度是 B*W/8*H/8,128 (把通道放在最后，用于传给线性模型)
        rpn_features = rpn_features_original.permute(0, 2, 3, 1).reshape(-1, self.features_channels)
        # 维度是 B,128,W/8,H/8
        cls_features = self.cls_resnet(x)

        # ***** 选取区域部分 *****
        # 根据区域特征生成各个锚点对应的对象可能性
        # 维度是 B,W/8*H/8*AnchorBoxes,2
        rpn_labels = self.rpn_labels_model(rpn_features)
        rpn_labels = rpn_labels.reshape(
            rpn_features_original.shape[0],
            rpn_features_original.shape[2] * rpn_features_original.shape[3] * MyModel.AnchorBoxes,
            2)
        # 根据区域特征生成各个锚点对应的区域偏移
        # 维度是 B,W/8*H/8*AnchorBoxes,4
        rpn_offsets = self.rpn_offsets_model(rpn_features)
        rpn_offsets = rpn_offsets.reshape(
            rpn_features_original.shape[0],
            rpn_features_original.shape[2] * rpn_features_original.shape[3] * MyModel.AnchorBoxes,
            4)
        # 选取可能出现对象的区域，并调整区域范围
        with torch.no_grad():
            rpn_scores = nn.functional.softmax(rpn_labels, dim=2)[:,:,1]
            # 选取可能性最高的部分区域
            rpn_top_scores = torch.topk(rpn_scores, k=self.rpn_max_candidates, dim=1)
            rpn_candidates_batch = []
            for x in range(0, rpn_scores.shape[0]):
                rpn_candidates = []
                for score, index in zip(rpn_top_scores.values[x], rpn_top_scores.indices[x]):
                    # 过滤可能性低于指定阈值的区域
                    if score.item() < self.rpn_score_threshold:
                        continue
                    anchor_box = MyModel.Anchors[index.item()]
                    offset = rpn_offsets[x,index.item()].tolist()
                    # 调整区域范围
                    candidate_box = adjust_box_by_offset(anchor_box, offset)
                    rpn_candidates.append(candidate_box)
                rpn_candidates_batch.append(rpn_candidates)

        # ***** 判断分类部分 *****
        cls_output = []
        cls_result = []
        for index in range(0, cls_features.shape[0]):
            pooled = MyModel._roi_crop(
                cls_features[index], rpn_candidates_batch[index], self.pooling_size)
            if pooled is None:
                # 没有找到可能包含对象的区域
                cls_output.append(None)
                cls_result.append(None)
                continue
            pooled = pooled.reshape(pooled.shape[0], -1)
            labels = self.cls_labels_model(pooled)
            offsets = self.cls_offsets_model(pooled)
            cls_output.append((labels, offsets))
            # 使用 softmax 判断可能性最大的分类
            classes = nn.functional.softmax(labels, dim=1).max(dim=1).indices
            # 根据分类对应的偏移再次调整区域范围
            offsets_map = offsets.reshape(offsets.shape[0] * len(CLASSES), 4)
            result = []
            for box_index in range(0, classes.shape[0]):
                predicted_label = classes[box_index].item()
                if predicted_label == 0:
                    continue # 0 代表 other, 表示非对象
                candidate_box = rpn_candidates_batch[index][box_index]
                offset = offsets_map[box_index * len(CLASSES) + predicted_label].tolist()
                predicted_box = adjust_box_by_offset(candidate_box, offset)
                # 添加分类与最终预测区域
                result.append((predicted_label, predicted_box))
            cls_result.append(result)

        # 前面的项目用于学习，最后一项是最终输出结果
        return rpn_labels, rpn_offsets, rpn_candidates_batch, cls_output, cls_result

    @staticmethod
    def loss_function(predicted, actual):
        """Faster-RCNN 使用的多任务损失计算器"""
        rpn_labels, rpn_offsets, rpn_candidates_batch, cls_output, _ = predicted
        rpn_labels_losses = []
        rpn_offsets_losses = []
        cls_labels_losses = []
        cls_offsets_losses = []
        for batch_index in range(len(actual)):
            # 计算 RPN 的损失
            (true_boxes_labels,
                actual_rpn_labels, actual_rpn_labels_mask,
                actual_rpn_offsets, actual_rpn_offsets_mask) = actual[batch_index]
            rpn_labels_losses.append(nn.functional.cross_entropy(
                rpn_labels[batch_index][actual_rpn_labels_mask],
                actual_rpn_labels.to(device)))
            rpn_offsets_losses.append(nn.functional.smooth_l1_loss(
                rpn_offsets[batch_index][actual_rpn_offsets_mask],
                actual_rpn_offsets.to(device)))
            # 计算标签分类的损失
            if cls_output[batch_index] is None:
                continue
            cls_labels_mask = []
            cls_offsets_mask = []
            cls_actual_labels = []
            cls_actual_offsets = []
            cls_predicted_labels, cls_predicted_offsets = cls_output[batch_index]
            cls_predicted_offsets_map = cls_predicted_offsets.reshape(-1, 4)
            rpn_candidates = rpn_candidates_batch[batch_index]
            for box_index, candidate_box in enumerate(rpn_candidates):
                iou_list = [ calc_iou(candidate_box, true_box) for (_, true_box) in true_boxes_labels ]
                positive_index = next((index for index, iou in enumerate(iou_list) if iou > IOU_POSITIVE_THRESHOLD), None)
                is_negative = all(iou < IOU_NEGATIVE_THRESHOLD for iou in iou_list)
                if positive_index is not None:
                    true_label, true_box = true_boxes_labels[positive_index]
                    cls_actual_labels.append(true_label)
                    cls_labels_mask.append(box_index)
                    # 如果区域正确，则学习真实分类对应的区域偏移
                    cls_actual_offsets.append(calc_box_offset(candidate_box, true_box))
                    cls_offsets_mask.append(box_index * len(CLASSES) + true_label)
                elif is_negative:
                    cls_actual_labels.append(0) # 0 代表 other, 表示非对象
                    cls_labels_mask.append(box_index)
                # 如果候选区域与真实区域的重叠率介于两个阈值之间，则不参与学习
            if cls_labels_mask:
                cls_labels_losses.append(nn.functional.cross_entropy(
                    cls_predicted_labels[cls_labels_mask],
                    torch.tensor(cls_actual_labels).to(device)))
            if cls_offsets_mask:
                cls_offsets_losses.append(nn.functional.smooth_l1_loss(
                    cls_predicted_offsets_map[cls_offsets_mask],
                    torch.tensor(cls_actual_offsets).to(device)))
        # 合并损失值
        # 注意 loss 不可以使用 += 合并
        loss = torch.tensor(.0, requires_grad=True)
        loss = loss + torch.mean(torch.stack(rpn_labels_losses))
        loss = loss + torch.mean(torch.stack(rpn_offsets_losses))
        if cls_labels_losses:
            loss = loss + torch.mean(torch.stack(cls_labels_losses))
        if cls_offsets_losses:
            loss = loss + torch.mean(torch.stack(cls_offsets_losses))
        return loss

    @staticmethod
    def calc_accuracy(actual, predicted):
        """Faster-RCNN 使用的正确率计算器，这里只计算 RPN 与标签分类的正确率，区域偏移不计算"""
        rpn_labels, rpn_offsets, rpn_candidates_batch, cls_output, cls_result = predicted
        rpn_acc = 0
        cls_acc = 0
        for batch_index in range(len(actual)):
            # 计算 RPN 的正确率，正样本和负样本的正确率分别计算再平均
            (true_boxes_labels,
                actual_rpn_labels, actual_rpn_labels_mask,
                actual_rpn_offsets, actual_rpn_offsets_mask) = actual[batch_index]
            a = actual_rpn_labels.to(device)
            p = torch.max(rpn_labels[batch_index][actual_rpn_labels_mask], 1).indices
            rpn_acc_positive = ((a == 0) & (p == 0)).sum().item() / ((a == 0).sum().item() + 0.00001)
            rpn_acc_negative = ((a == 1) & (p == 1)).sum().item() / ((a == 1).sum().item() + 0.00001)
            rpn_acc += (rpn_acc_positive + rpn_acc_negative) / 2
            # 计算标签分类的正确率
            # 正确率 = 有对应预测区域并且预测分类正确的真实区域数量 / 总真实区域数量
            cls_correct = 0
            for true_label, true_box in true_boxes_labels:
                if cls_result[batch_index] is None:
                    continue
                for predicted_label, predicted_box in cls_result[batch_index]:
                    if calc_iou(predicted_box, true_box) > IOU_POSITIVE_THRESHOLD and predicted_label == true_label:
                        cls_correct += 1
                        break
            cls_acc += cls_correct / len(true_boxes_labels)
        rpn_acc /= len(actual)
        cls_acc /= len(actual)
        return rpn_acc, cls_acc

MyModel.Anchors = MyModel._generate_anchors(8)

def save_tensor(tensor, path):
    """保存 tensor 对象到文件"""
    torch.save(tensor, gzip.GzipFile(path, "wb"))

def load_tensor(path):
    """从文件读取 tensor 对象"""
    return torch.load(gzip.GzipFile(path, "rb"))

def calc_resize_parameters(sw, sh):
    """计算缩放图片的参数"""
    sw_new, sh_new = sw, sh
    dw, dh = IMAGE_SIZE
    pad_w, pad_h = 0, 0
    if sw / sh < dw / dh:
        sw_new = int(dw / dh * sh)
        pad_w = (sw_new - sw) // 2 # 填充左右
    else:
        sh_new = int(dh / dw * sw)
        pad_h = (sh_new - sh) // 2 # 填充上下
    return sw_new, sh_new, pad_w, pad_h

def resize_image(img):
    """缩放图片，比例不一致时填充"""
    sw, sh = img.size
    sw_new, sh_new, pad_w, pad_h = calc_resize_parameters(sw, sh)
    img_new = Image.new("RGB", (sw_new, sh_new))
    img_new.paste(img, (pad_w, pad_h))
    img_new = img_new.resize(IMAGE_SIZE)
    return img_new

def image_to_tensor(img):
    """转换图片对象到 tensor 对象"""
    arr = numpy.asarray(img)
    t = torch.from_numpy(arr)
    t = t.transpose(0, 2) # 转换维度 H,W,C 到 C,W,H
    t = t / 255.0 # 正规化数值使得范围在 0 ~ 1
    return t

def map_box_to_resized_image(box, sw, sh):
    """把原始区域转换到缩放后的图片对应的区域"""
    x, y, w, h = box
    sw_new, sh_new, pad_w, pad_h = calc_resize_parameters(sw, sh)
    scale = IMAGE_SIZE[0] / sw_new
    x = int((x + pad_w) * scale)
    y = int((y + pad_h) * scale)
    w = int(w * scale)
    h = int(h * scale)
    if x + w > IMAGE_SIZE[0] or y + h > IMAGE_SIZE[1] or w == 0 or h == 0:
        return 0, 0, 0, 0
    return x, y, w, h

def map_box_to_original_image(box, sw, sh):
    """把缩放后图片对应的区域转换到缩放前的原始区域"""
    x, y, w, h = box
    sw_new, sh_new, pad_w, pad_h = calc_resize_parameters(sw, sh)
    scale = IMAGE_SIZE[0] / sw_new
    x = int(x / scale - pad_w)
    y = int(y / scale - pad_h)
    w = int(w / scale)
    h = int(h / scale)
    if x + w > sw or y + h > sh or x < 0 or y < 0 or w == 0 or h == 0:
        return 0, 0, 0, 0
    return x, y, w, h

def calc_iou(rect1, rect2):
    """计算两个区域重叠部分 / 合并部分的比率 (intersection over union)"""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    xi = max(x1, x2)
    yi = max(y1, y2)
    wi = min(x1+w1, x2+w2) - xi
    hi = min(y1+h1, y2+h2) - yi
    if wi > 0 and hi > 0: # 有重叠部分
        area_overlap = wi*hi
        area_all = w1*h1 + w2*h2 - area_overlap
        iou = area_overlap / area_all
    else: # 没有重叠部分
        iou = 0
    return iou

def calc_box_offset(candidate_box, true_box):
    """计算候选区域与实际区域的偏移值"""
    # 这里计算出来的偏移值基于比例，而不受具体位置和大小影响
    # w h 使用 log 是为了减少过大的值的影响
    x1, y1, w1, h1 = candidate_box
    x2, y2, w2, h2 = true_box
    x_offset = (x2 - x1) / w1
    y_offset = (y2 - y1) / h1
    w_offset = math.log(w2 / w1)
    h_offset = math.log(h2 / h1)
    return (x_offset, y_offset, w_offset, h_offset)

def adjust_box_by_offset(candidate_box, offset):
    """根据偏移值调整候选区域"""
    x1, y1, w1, h1 = candidate_box
    x_offset, y_offset, w_offset, h_offset = offset
    x2 = min(IMAGE_SIZE[0]-1,  max(0, w1 * x_offset + x1))
    y2 = min(IMAGE_SIZE[1]-1,  max(0, h1 * y_offset + y1))
    w2 = min(IMAGE_SIZE[0]-x2, max(1, math.exp(w_offset) * w1))
    h2 = min(IMAGE_SIZE[1]-y2, max(1, math.exp(h_offset) * h1))
    return (x2, y2, w2, h2)

def merge_box(box_a, box_b):
    """合并两个区域"""
    x1, y1, w1, h1 = box_a
    x2, y2, w2, h2 = box_b
    x = min(x1, x2)
    y = min(y1, y2)
    w = max(x1 + w1, x2 + w2) - x
    h = max(y1 + h1, y2 + h2) - y
    return (x, y, w, h)

def prepare_save_batch(batch, image_tensors, image_boxes_labels):
    """准备训练 - 保存单个批次的数据"""
    # 按索引值列表生成输入和输出 tensor 对象的函数
    def split_dataset(indices):
        image_in = []
        boxes_labels_out = {}
        for new_image_index, original_image_index in enumerate(indices.tolist()):
            image_in.append(image_tensors[original_image_index])
            boxes_labels_out[new_image_index] = image_boxes_labels[original_image_index]
        tensor_image_in = torch.stack(image_in) # 维度: B,C,W,H
        return tensor_image_in, boxes_labels_out

    # 切分训练集 (80%)，验证集 (10%) 和测试集 (10%)
    random_indices = torch.randperm(len(image_tensors))
    training_indices = random_indices[:int(len(random_indices)*0.8)]
    validating_indices = random_indices[int(len(random_indices)*0.8):int(len(random_indices)*0.9):]
    testing_indices = random_indices[int(len(random_indices)*0.9):]
    training_set = split_dataset(training_indices)
    validating_set = split_dataset(validating_indices)
    testing_set = split_dataset(testing_indices)

    # 保存到硬盘
    save_tensor(training_set, f"data/training_set.{batch}.pt")
    save_tensor(validating_set, f"data/validating_set.{batch}.pt")
    save_tensor(testing_set, f"data/testing_set.{batch}.pt")
    print(f"batch {batch} saved")

def prepare():
    """准备训练"""
    # 数据集转换到 tensor 以后会保存在 data 文件夹下
    if not os.path.isdir("data"):
        os.makedirs("data")

    # 加载图片和图片对应的区域与分类列表
    # { 图片名: [ 区域与分类, 区域与分类, .. ] }
    box_map = defaultdict(lambda: [])
    for filename in os.listdir(IMAGE_DIR):
        xml_path = os.path.join(ANNOTATION_DIR, filename.split(".")[0] + ".xml")
        if not os.path.isfile(xml_path):
            continue
        tree = ET.ElementTree(file=xml_path)
        objects = tree.findall("object")
        for obj in objects:
            class_name = obj.find("name").text
            x1 = int(obj.find("bndbox/xmin").text)
            x2 = int(obj.find("bndbox/xmax").text)
            y1 = int(obj.find("bndbox/ymin").text)
            y2 = int(obj.find("bndbox/ymax").text)
            box_map[filename].append((x1, y1, x2-x1, y2-y1, CLASSES_MAPPING[class_name]))

    # 保存图片和图片对应的分类与区域列表
    batch_size = 20
    batch = 0
    image_tensors = [] # 图片列表
    image_boxes_labels = {} # 图片对应的真实区域与分类列表，和候选区域与区域偏移
    for filename, original_boxes_labels in box_map.items():
        image_path = os.path.join(IMAGE_DIR, filename)
        with Image.open(image_path) as img_original: # 加载原始图片
            sw, sh = img_original.size # 原始图片大小
            img = resize_image(img_original) # 缩放图片
            image_index = len(image_tensors) # 图片在批次中的索引值
            image_tensors.append(image_to_tensor(img)) # 添加图片到列表
            true_boxes_labels = [] # 图片对应的真实区域与分类列表
        # 添加真实区域与分类列表
        for box_label in original_boxes_labels:
            x, y, w, h, label = box_label
            x, y, w, h = map_box_to_resized_image((x, y, w, h), sw, sh) # 缩放实际区域
            if w < 10 or h < 10:
                continue # 缩放后区域过小
            # 检查计算是否有问题
            # child_img = img.copy().crop((x, y, x+w, y+h))
            # child_img.save(f"{filename}_{x}_{y}_{w}_{h}_{label}.png")
            true_boxes_labels.append((label, (x, y, w, h)))
        # 如果图片中的所有区域都过小则跳过
        if not true_boxes_labels:
            image_tensors.pop()
            image_index = len(image_tensors)
            continue
        # 根据锚点列表寻找候选区域，并计算区域偏移
        actual_rpn_labels = []
        actual_rpn_labels_mask = []
        actual_rpn_offsets = []
        actual_rpn_offsets_mask = []
        positive_index_set = set()
        for index, anchor_box in enumerate(MyModel.Anchors):
            # 如果候选区域和任意一个实际区域重叠率大于阈值，则认为是正样本
            # 如果候选区域和所有实际区域重叠率都小于阈值，则认为是负样本
            # 重叠率介于两个阈值之间的区域不参与学习
            iou_list = [ calc_iou(anchor_box, true_box) for (_, true_box) in true_boxes_labels ]
            positive_index = next((index for index, iou in enumerate(iou_list) if iou > IOU_POSITIVE_THRESHOLD), None)
            is_negative = all(iou < IOU_NEGATIVE_THRESHOLD for iou in iou_list)
            if positive_index is not None:
                positive_index_set.add(positive_index)
                actual_rpn_labels.append(1)
                actual_rpn_labels_mask.append(index)
                # 只有包含对象的区域参需要调整偏移
                true_box = true_boxes_labels[positive_index][1]
                actual_rpn_offsets.append(calc_box_offset(anchor_box, true_box))
                actual_rpn_offsets_mask.append(index)
            elif is_negative:
                actual_rpn_labels.append(0)
                actual_rpn_labels_mask.append(index)
        # 输出找不到候选区域的真实区域，调整锚点生成参数时使用
        # for index in range(len(true_boxes_labels)):
        #    if index not in positive_index_set:
        #        print(true_boxes_labels[index][1])
        # print("-----")
        # 如果一个候选区域都找不到则跳过
        if not positive_index_set:
            image_tensors.pop()
            image_index = len(image_tensors)
            continue
        image_boxes_labels[image_index] = (
            true_boxes_labels,
            torch.tensor(actual_rpn_labels, dtype=torch.long),
            torch.tensor(actual_rpn_labels_mask, dtype=torch.long),
            torch.tensor(actual_rpn_offsets, dtype=torch.float),
            torch.tensor(actual_rpn_offsets_mask, dtype=torch.long))
        # 保存批次
        if len(image_tensors) >= batch_size:
            prepare_save_batch(batch, image_tensors, image_boxes_labels)
            image_tensors.clear()
            image_boxes_labels.clear()
            batch += 1
    # 保存剩余的批次
    if len(image_tensors) > 10:
        prepare_save_batch(batch, image_tensors, image_boxes_labels)

def train():
    """开始训练"""
    # 创建模型实例
    model = MyModel().to(device)

    # 创建多任务损失计算器
    loss_function = MyModel.loss_function

    # 创建参数调整器
    optimizer = torch.optim.Adam(model.parameters())

    # 记录训练集和验证集的正确率变化
    training_rpn_accuracy_history = []
    training_cls_accuracy_history = []
    validating_rpn_accuracy_history = []
    validating_cls_accuracy_history = []

    # 记录最高的验证集正确率
    validating_rpn_accuracy_highest = -1
    validating_rpn_accuracy_highest_epoch = 0
    validating_cls_accuracy_highest = -1
    validating_cls_accuracy_highest_epoch = 0

    # 读取批次的工具函数
    def read_batches(base_path):
        for batch in itertools.count():
            path = f"{base_path}.{batch}.pt"
            if not os.path.isfile(path):
                break
            x, y = load_tensor(path)
            yield x.to(device), y

    # 计算正确率的工具函数
    calc_accuracy = MyModel.calc_accuracy

    # 开始训练过程
    for epoch in range(1, 10000):
        print(f"epoch: {epoch}")

        # 根据训练集训练并修改参数
        # 切换模型到训练模式，将会启用自动微分，批次正规化 (BatchNorm) 与 Dropout
        model.train()
        training_rpn_accuracy_list = []
        training_cls_accuracy_list = []
        for batch_index, batch in enumerate(read_batches("data/training_set")):
            # 划分输入和输出
            batch_x, batch_y = batch
            # 计算预测值
            predicted = model(batch_x)
            # 计算损失
            loss = loss_function(predicted, batch_y)
            # 从损失自动微分求导函数值
            loss.backward()
            # 使用参数调整器调整参数
            optimizer.step()
            # 清空导函数值
            optimizer.zero_grad()
            # 记录这一个批次的正确率，torch.no_grad 代表临时禁用自动微分功能
            with torch.no_grad():
                training_batch_rpn_accuracy, training_batch_cls_accuracy = calc_accuracy(batch_y, predicted)
            # 输出批次正确率
            training_rpn_accuracy_list.append(training_batch_rpn_accuracy)
            training_cls_accuracy_list.append(training_batch_cls_accuracy)
            print(f"epoch: {epoch}, batch: {batch_index}: " +
                f"batch rpn accuracy: {training_batch_rpn_accuracy}, cls accuracy: {training_batch_cls_accuracy}")
        training_rpn_accuracy = sum(training_rpn_accuracy_list) / len(training_rpn_accuracy_list)
        training_cls_accuracy = sum(training_cls_accuracy_list) / len(training_cls_accuracy_list)
        training_rpn_accuracy_history.append(training_rpn_accuracy)
        training_cls_accuracy_history.append(training_cls_accuracy)
        print(f"training rpn accuracy: {training_rpn_accuracy}, cls accuracy: {training_cls_accuracy}")

        # 检查验证集
        # 切换模型到验证模式，将会禁用自动微分，批次正规化 (BatchNorm) 与 Dropout
        model.eval()
        validating_rpn_accuracy_list = []
        validating_cls_accuracy_list = []
        for batch in read_batches("data/validating_set"):
            batch_x, batch_y = batch
            predicted = model(batch_x)
            validating_batch_rpn_accuracy, validating_batch_cls_accuracy = calc_accuracy(batch_y, predicted)
            validating_rpn_accuracy_list.append(validating_batch_rpn_accuracy)
            validating_cls_accuracy_list.append(validating_batch_cls_accuracy)
        validating_rpn_accuracy = sum(validating_rpn_accuracy_list) / len(validating_rpn_accuracy_list)
        validating_cls_accuracy = sum(validating_cls_accuracy_list) / len(validating_cls_accuracy_list)
        validating_rpn_accuracy_history.append(validating_rpn_accuracy)
        validating_cls_accuracy_history.append(validating_cls_accuracy)
        print(f"validating rpn accuracy: {validating_rpn_accuracy}, cls accuracy: {validating_cls_accuracy}")

        # 记录最高的验证集正确率与当时的模型状态，判断是否在 20 次训练后仍然没有刷新记录
        if validating_rpn_accuracy > validating_rpn_accuracy_highest:
            validating_rpn_accuracy_highest = validating_rpn_accuracy
            validating_rpn_accuracy_highest_epoch = epoch
            save_tensor(model.state_dict(), "model.pt")
            print("highest rpn validating accuracy updated")
        elif validating_cls_accuracy > validating_cls_accuracy_highest:
            validating_cls_accuracy_highest = validating_cls_accuracy
            validating_cls_accuracy_highest_epoch = epoch
            save_tensor(model.state_dict(), "model.pt")
            print("highest cls validating accuracy updated")
        elif (epoch - validating_rpn_accuracy_highest_epoch > 20 and
            epoch - validating_cls_accuracy_highest_epoch > 20):
            # 在 20 次训练后仍然没有刷新记录，结束训练
            print("stop training because highest validating accuracy not updated in 20 epoches")
            break

    # 使用达到最高正确率时的模型状态
    print(f"highest rpn validating accuracy: {validating_rpn_accuracy_highest}",
        f"from epoch {validating_rpn_accuracy_highest_epoch}")
    print(f"highest cls validating accuracy: {validating_cls_accuracy_highest}",
        f"from epoch {validating_cls_accuracy_highest_epoch}")
    model.load_state_dict(load_tensor("model.pt"))

    # 检查测试集
    testing_rpn_accuracy_list = []
    testing_cls_accuracy_list = []
    for batch in read_batches("data/testing_set"):
        batch_x, batch_y = batch
        predicted = model(batch_x)
        testing_batch_rpn_accuracy, testing_batch_cls_accuracy = calc_accuracy(batch_y, predicted)
        testing_rpn_accuracy_list.append(testing_batch_rpn_accuracy)
        testing_cls_accuracy_list.append(testing_batch_cls_accuracy)
    testing_rpn_accuracy = sum(testing_rpn_accuracy_list) / len(testing_rpn_accuracy_list)
    testing_cls_accuracy = sum(testing_cls_accuracy_list) / len(testing_cls_accuracy_list)
    print(f"testing rpn accuracy: {testing_rpn_accuracy}, cls accuracy: {testing_cls_accuracy}")

    # 显示训练集和验证集的正确率变化
    pyplot.plot(training_rpn_accuracy_history, label="training_rpn_accuracy")
    pyplot.plot(training_cls_accuracy_history, label="training_cls_accuracy")
    pyplot.plot(validating_rpn_accuracy_history, label="validating_rpn_accuracy")
    pyplot.plot(validating_cls_accuracy_history, label="validating_cls_accuracy")
    pyplot.ylim(0, 1)
    pyplot.legend()
    pyplot.show()

def eval_model():
    """使用训练好的模型"""
    # 创建模型实例，加载训练好的状态，然后切换到验证模式
    model = MyModel().to(device)
    model.load_state_dict(load_tensor("model.pt"))
    model.eval()

    # 询问图片路径，并显示所有可能是人脸的区域
    while True:
        try:
            image_path = input("Image path: ")
            if not image_path:
                continue
            # 构建输入
            with Image.open(image_path) as img_original: # 加载原始图片
                sw, sh = img_original.size # 原始图片大小
                img = resize_image(img_original) # 缩放图片
                img_output = img_original.copy() # 复制图片，用于后面添加标记
                tensor_in = image_to_tensor(img)
            # 预测输出
            cls_result = model(tensor_in.unsqueeze(0).to(device))[-1][0]
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
            for label, box in final_result:
                x, y, w, h = map_box_to_original_image(box, sw, sh)
                draw.rectangle((x, y, x+w, y+h), outline="#FF0000")
                draw.text((x, y-10), CLASSES[label], fill="#FF0000")
                print((x, y, w, h), CLASSES[label])
            img_output.save("img_output.png")
            print("saved to img_output.png")
            print()
        except Exception as e:
            print("error:", e)

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print(f"Please run: {sys.argv[0]} prepare|train|eval")
        exit()

    # 给随机数生成器分配一个初始值，使得每次运行都可以生成相同的随机数
    # 这是为了让过程可重现，你也可以选择不这样做
    random.seed(0)
    torch.random.manual_seed(0)

    # 根据命令行参数选择操作
    operation = sys.argv[1]
    if operation == "prepare":
        prepare()
    elif operation == "train":
        train()
    elif operation == "eval":
        eval_model()
    else:
        raise ValueError(f"Unsupported operation: {operation}")

if __name__ == "__main__":
    main()