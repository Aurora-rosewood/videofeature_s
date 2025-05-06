from pyexpat import features
from torchvision.models import ResNet18_Weights,ResNet50_Weights,ResNet101_Weights,ResNet152_Weights, ResNet200_Weights, VGG16_Weights, DenseNet121_Weights, Inception_V3_Weights
from sklearn.decomposition import PCA
import matplotlib.animation as animation
from sklearn.manifold import TSNE
from matplotlib.animation import FuncAnimation
import seaborn as sns
import tkinter as tk
from tkinter import font, Label, Button, ttk
import threading
from tkinter import filedialog, Tk, Label, Button, ttk
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import Inception_V3_Weights
import torch.nn as nn
import torchvision.models as models
import cv2
from torchvision import transforms
from tkinter import filedialog
import torch
import PySide6
import PySide6.QtWidgets as QtWidgets
import PySide6.QtCore as QtCore
import PySide6.QtGui as QtGui


video_features = []  # 全局变量，用于存储视频特征


#resnet18
def use_extract_features_CNN_resnet18():  # 选择视频
    # 打开文件选择对话框让用户选择视频文件
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:  # 确保用户选择了文件
        threading.Thread(target=extract_and_display_features_resnet18, args=(video_path,)).start()

def use_extract_features1_CNN_resnet18():  # 选择视频
    # 打开文件选择对话框让用户选择视频文件
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:  # 确保用户选择了文件
        threading.Thread(target=extract_and_display_features1_resnet18, args=(video_path,)).start()

def use_extract_features2_CNN_resnet18():  # 选择视频
    # 打开文件选择对话框让用户选择视频文件
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:  # 确保用户选择了文件
        threading.Thread(target=extract_and_display_features2_resnet18, args=(video_path,)).start()

def use_extract_features3_CNN_resnet18():  # 选择视频
    # 打开文件选择对话框让用户选择视频文件
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:  # 确保用户选择了文件
        threading.Thread(target=extract_and_display_features3_resnet18, args=(video_path,)).start()

def extract_and_display_features_resnet18(video_path): #更新全局变量并显示 调用模型
    global video_features
    video_features = extract_features_CNN_resnet18(video_path)  # 提取特征并更新全局变量
    result_label.config(text=f"resnet18特征提取完成，特征长度：{len(video_features)}")  # 更新标签内容显示特征长度
    print(video_path)
    # print(video_features)

def extract_and_display_features1_resnet18(video_path): #更新全局变量并显示 调用模型
    global video_features
    video_features = extract_features1_CNN_resnet18(video_path)  # 提取特征并更新全局变量
    result_label.config(text=f"resnet18特征提取完成，特征长度：{len(video_features)}")  # 更新标签内容显示特征长度
    print(video_path)
    # print(video_features)

def extract_and_display_features2_resnet18(video_path): #更新全局变量并显示 调用模型
    global video_features
    video_features = extract_features2_CNN_resnet18(video_path)  # 提取特征并更新全局变量
    result_label.config(text=f"resnet18特征提取完成，特征长度：{len(video_features)}")  # 更新标签内容显示特征长度
    print(video_path)
    # print(video_features)

def extract_and_display_features3_resnet18(video_path): #更新全局变量并显示 调用模型
    global video_features
    video_features = extract_features3_CNN_resnet18(video_path)  # 提取特征并更新全局变量
    result_label.config(text=f"resnet18特征提取完成，特征长度：{len(video_features)}")  # 更新标签内容显示特征长度
    print(video_path)
    # print(video_features)

#resnet18
def extract_features_CNN_resnet18(video_path):  # CNN_resnet18模型提取特征
    # 使用更新的权重选择方式初始化ResNet模型
    weights = ResNet18_Weights.DEFAULT  # ResNet18默认权重配置
    model = models.resnet18(weights=weights)  # 加载预训练的ResNet模型
    model.eval()  # 设置为评估模式

    if torch.cuda.is_available():
        model = model.to('cuda')  # 判断当前环境是否可以使用GPU

    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),  # 缩放为256*256
        transforms.CenterCrop(224),  # 中心剪裁为224*224
        transforms.ToTensor(),  # 将裁剪后的图像转换为张量格式，以便模型能够处理
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])  # 对张量进行归一化处理，使用指定的均值和标准差

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while True:
        ret, frame = cap.read()  # ret 成功显示成功与否  frame 读取到的帧
        if not ret:
            break

        # 应用预处理并添加批次维度
        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            output = model(input_batch)

        # 将特征转换为一维列表并存储
        features.append(output.cpu().numpy().flatten())

        # 更新进度条
        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    cap.release()
    return features

def dimension_reduction_pca_resnet18(video_features): #实现了利用主成分分析（PCA）来降低视频特征的维度，并且使用散点图来可视化降维后的结果
    # 假设 features 是你所有帧的特征列表，每个元素是一个特征向量

    features = np.array(video_features) #将视频特征列表转为数组
    # 应用PCA
    pca = PCA(n_components=2)  # 降至2维
    reduced_features = pca.fit_transform(video_features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

# 第一层
def extract_features1_CNN_resnet18(video_path):
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()

    if torch.cuda.is_available():
        model = model.to('cuda')

    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    features1 = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # 定义一个钩子函数来捕获第一层的输出
    layer_features1 = None

    def hook(module, input, output):
        nonlocal layer_features1
        layer_features1 = output

    model.conv1.register_forward_hook(hook)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            model(input_batch)

        # 将第一层特征转换为一维列表并存储
        if layer_features1 is not None:
            features1.append(layer_features1.cpu().numpy().flatten())

        # 更新进度条
        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    cap.release()
    return features1

def dimension_reduction_pca1_resnet18(video_features): #实现了利用主成分分析（PCA）来降低视频特征的维度，并且使用散点图来可视化降维后的结果
    # 假设 features 是你所有帧的特征列表，每个元素是一个特征向量

    features1 = np.array(video_features) #将视频特征列表转为数组
    # 应用PCA
    pca = PCA(n_components=2)  # 降至2维
    reduced_features = pca.fit_transform(features1)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

# 第二层
def extract_features2_CNN_resnet18(video_path):
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()

    if torch.cuda.is_available():
        model = model.to('cuda')

    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    features2 = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # 定义一个钩子函数来捕获第一层的输出
    layer_features2 = None

    def hook(module, input, output):
        nonlocal layer_features2
        layer_features2 = output

    model.layer1.register_forward_hook(hook)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            model(input_batch)

        # 将第一层特征转换为一维列表并存储
        if layer_features2 is not None:
            features2.append(layer_features2.cpu().numpy().flatten())

        # 更新进度条
        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    cap.release()
    return features2

def dimension_reduction_pca2_resnet18(video_features): #实现了利用主成分分析（PCA）来降低视频特征的维度，并且使用散点图来可视化降维后的结果
    # 假设 features 是你所有帧的特征列表，每个元素是一个特征向量

    features2 = np.array(video_features) #将视频特征列表转为数组
    # 应用PCA
    pca = PCA(n_components=2)  # 降至2维
    reduced_features = pca.fit_transform(features2)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

# 第三层
def extract_features3_CNN_resnet18(video_path):
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()

    if torch.cuda.is_available():
        model = model.to('cuda')

    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    features3 = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # 定义一个钩子函数来捕获第一层的输出
    layer_features3 = None

    def hook(module, input, output):
        nonlocal layer_features3
        layer_features3 = output

    model.layer2.register_forward_hook(hook)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            model(input_batch)

        # 将第一层特征转换为一维列表并存储
        if layer_features3 is not None:
            features3.append(layer_features3.cpu().numpy().flatten())

        # 更新进度条
        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    cap.release()
    return features3

def dimension_reduction_pca3_resnet18(video_features): #实现了利用主成分分析（PCA）来降低视频特征的维度，并且使用散点图来可视化降维后的结果
    # 假设 features 是你所有帧的特征列表，每个元素是一个特征向量

    features3 = np.array(video_features) #将视频特征列表转为数组
    # 应用PCA
    pca = PCA(n_components=2)  # 降至2维
    reduced_features = pca.fit_transform(features3)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()


def feacture_change_dynamics_resnet18(video_features):
    # 准备画布
    fig, ax = plt.subplots()
    frame_features = video_features[0].reshape((25, 40))
    img = ax.imshow(frame_features, cmap='viridis')

    def update(frame_number):
        frame_features = video_features[frame_number].reshape((25, 40))
        img.set_data(frame_features)
        ax.set_title(f"Frame {frame_number + 1}")
        return img,

    # 创建动画
    # ani = FuncAnimation(fig, update, frames=278, interval=50)  # 每帧间隔50毫秒
    ani = FuncAnimation(fig, update, frames=len(video_features), interval=50)
    plt.show()



#resnet50
def use_extract_features_CNN_resnet50():  # 选择视频
    # 打开文件选择对话框让用户选择视频文件
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:  # 确保用户选择了文件
        threading.Thread(target=extract_and_display_features_resnet50, args=(video_path,)).start()

def use_extract_features1_CNN_resnet50():  # 选择视频
    # 打开文件选择对话框让用户选择视频文件
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:  # 确保用户选择了文件
        threading.Thread(target=extract_and_display_features1_resnet50, args=(video_path,)).start()

def use_extract_features2_CNN_resnet50():  # 选择视频
    # 打开文件选择对话框让用户选择视频文件
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:  # 确保用户选择了文件
        threading.Thread(target=extract_and_display_features2_resnet50, args=(video_path,)).start()

def use_extract_features3_CNN_resnet50():  # 选择视频
    # 打开文件选择对话框让用户选择视频文件
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:  # 确保用户选择了文件
        threading.Thread(target=extract_and_display_features3_resnet50, args=(video_path,)).start()

def extract_and_display_features_resnet50(video_path): #更新全局变量并显示 调用模型
    global video_features
    video_features = extract_features_CNN_resnet50(video_path)  # 提取特征并更新全局变量
    result_label.config(text=f"resnet50特征提取完成，特征长度：{len(video_features)}")  # 更新标签内容显示特征长度
    print(video_path)
    # print(video_features)

def extract_and_display_features1_resnet50(video_path): #更新全局变量并显示 调用模型
    global video_features
    video_features = extract_features1_CNN_resnet50(video_path)  # 提取特征并更新全局变量
    result_label.config(text=f"resnet50特征提取完成，特征长度：{len(video_features)}")  # 更新标签内容显示特征长度
    print(video_path)
    # print(video_features)

def extract_and_display_features2_resnet50(video_path): #更新全局变量并显示 调用模型
    global video_features
    video_features = extract_features2_CNN_resnet50(video_path)  # 提取特征并更新全局变量
    result_label.config(text=f"resnet50特征提取完成，特征长度：{len(video_features)}")  # 更新标签内容显示特征长度
    print(video_path)
    # print(video_features)

def extract_and_display_features3_resnet50(video_path): #更新全局变量并显示 调用模型
    global video_features
    video_features = extract_features3_CNN_resnet50(video_path)  # 提取特征并更新全局变量
    result_label.config(text=f"resnet50特征提取完成，特征长度：{len(video_features)}")  # 更新标签内容显示特征长度
    print(video_path)
    # print(video_features)

def extract_features_CNN_resnet50(video_path):  # CNN_resnet50模型提取特征
    # 使用更新的权重选择方式初始化ResNet模型
    weights = ResNet50_Weights.DEFAULT  # ResNet50默认权重配置
    model = models.resnet50(weights=weights)  # 加载预训练的ResNet模型
    model.eval()  # 设置为评估模式

    if torch.cuda.is_available():
        model = model.to('cuda')  # 判断当前环境是否可以使用GPU

    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),  # 缩放为256*256
        transforms.CenterCrop(224),  # 中心剪裁为224*224
        transforms.ToTensor(),  # 将裁剪后的图像转换为张量格式，以便模型能够处理
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])  # 对张量进行归一化处理，使用指定的均值和标准差

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while True:
        ret, frame = cap.read()  # ret 成功显示成功与否  frame 读取到的帧
        if not ret:
            break

        # 应用预处理并添加批次维度
        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            output = model(input_batch)

        # 将特征转换为一维列表并存储
        features.append(output.cpu().numpy().flatten())

        # 更新进度条
        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    cap.release()
    return features

def dimension_reduction_pca_resnet50(video_features): #实现了利用主成分分析（PCA）来降低视频特征的维度，并且使用散点图来可视化降维后的结果
    # 假设 features 是你所有帧的特征列表，每个元素是一个特征向量

    features = np.array(video_features) #将视频特征列表转为数组
    # 应用PCA
    pca = PCA(n_components=2)  # 降至2维
    reduced_features = pca.fit_transform(video_features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

# 第一层
def extract_features1_CNN_resnet50(video_path):
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()

    if torch.cuda.is_available():
        model = model.to('cuda')

    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    features1 = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # 定义一个钩子函数来捕获第一层的输出
    layer_features1 = None

    def hook(module, input, output):
        nonlocal layer_features1
        layer_features1 = output

    model.conv1.register_forward_hook(hook)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            model(input_batch)

        # 将第一层特征转换为一维列表并存储
        if layer_features1 is not None:
            features1.append(layer_features1.cpu().numpy().flatten())

        # 更新进度条
        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    cap.release()
    return features1

def dimension_reduction_pca1_resnet50(video_features): #实现了利用主成分分析（PCA）来降低视频特征的维度，并且使用散点图来可视化降维后的结果
    # 假设 features 是你所有帧的特征列表，每个元素是一个特征向量

    features1 = np.array(video_features) #将视频特征列表转为数组
    # 应用PCA
    pca = PCA(n_components=2)  # 降至2维
    reduced_features = pca.fit_transform(video_features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

# 第二层
def extract_features2_CNN_resnet50(video_path):
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()

    if torch.cuda.is_available():
        model = model.to('cuda')

    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    features2 = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # 定义一个钩子函数来捕获第一层的输出
    layer_features2 = None

    def hook(module, input, output):
        nonlocal layer_features2
        layer_features2 = output

    model.layer1.register_forward_hook(hook)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            model(input_batch)

        # 将第一层特征转换为一维列表并存储
        if layer_features2 is not None:
            features2.append(layer_features2.cpu().numpy().flatten())

        # 更新进度条
        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    cap.release()
    return features2

def dimension_reduction_pca2_resnet50(video_features): #实现了利用主成分分析（PCA）来降低视频特征的维度，并且使用散点图来可视化降维后的结果
    # 假设 features 是你所有帧的特征列表，每个元素是一个特征向量

    features2 = np.array(video_features) #将视频特征列表转为数组
    # 应用PCA
    pca = PCA(n_components=2)  # 降至2维
    reduced_features = pca.fit_transform(video_features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

# 第三层
def extract_features3_CNN_resnet50(video_path):
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()

    if torch.cuda.is_available():
        model = model.to('cuda')

    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    features3 = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # 定义一个钩子函数来捕获第一层的输出
    layer_features3 = None

    def hook(module, input, output):
        nonlocal layer_features3
        layer_features3 = output

    model.layer2.register_forward_hook(hook)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            model(input_batch)

        # 将第一层特征转换为一维列表并存储
        if layer_features3 is not None:
            features3.append(layer_features3.cpu().numpy().flatten())

        # 更新进度条
        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    cap.release()
    return features3

def dimension_reduction_pca3_resnet50(video_features): #实现了利用主成分分析（PCA）来降低视频特征的维度，并且使用散点图来可视化降维后的结果
    # 假设 features 是你所有帧的特征列表，每个元素是一个特征向量

    features3 = np.array(video_features) #将视频特征列表转为数组
    # 应用PCA
    pca = PCA(n_components=2)  # 降至2维
    reduced_features = pca.fit_transform(video_features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()


def feacture_change_dynamics_resnet50(video_features):
    # 准备画布
    fig, ax = plt.subplots()
    frame_features = video_features[0].reshape((25, 40))
    img = ax.imshow(frame_features, cmap='viridis')

    def update(frame_number):
        frame_features = video_features[frame_number].reshape((25, 40))
        img.set_data(frame_features)
        ax.set_title(f"Frame {frame_number + 1}")
        return img,

    # 创建动画
    # ani = FuncAnimation(fig, update, frames=278, interval=50)  # 每帧间隔50毫秒
    ani = FuncAnimation(fig, update, frames=len(video_features), interval=50)
    plt.show()


#resnet101
def use_extract_features_CNN_resnet101():  # 选择视频
    # 打开文件选择对话框让用户选择视频文件
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:  # 确保用户选择了文件
        threading.Thread(target=extract_and_display_features_resnet101, args=(video_path,)).start()

def use_extract_features1_CNN_resnet101():  # 选择视频
    # 打开文件选择对话框让用户选择视频文件
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:  # 确保用户选择了文件
        threading.Thread(target=extract_and_display_features1_resnet101, args=(video_path,)).start()

def use_extract_features2_CNN_resnet101():  # 选择视频
    # 打开文件选择对话框让用户选择视频文件
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:  # 确保用户选择了文件
        threading.Thread(target=extract_and_display_features2_resnet101, args=(video_path,)).start()

def use_extract_features3_CNN_resnet101():  # 选择视频
    # 打开文件选择对话框让用户选择视频文件
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:  # 确保用户选择了文件
        threading.Thread(target=extract_and_display_features3_resnet101, args=(video_path,)).start()

def extract_and_display_features_resnet101(video_path): #更新全局变量并显示 调用模型
    global video_features
    video_features = extract_features_CNN_resnet101(video_path)  # 提取特征并更新全局变量
    result_label.config(text=f"resnet101特征提取完成，特征长度：{len(video_features)}")  # 更新标签内容显示特征长度
    print(video_path)
    # print(video_features)

def extract_and_display_features1_resnet101(video_path): #更新全局变量并显示 调用模型
    global video_features
    video_features = extract_features1_CNN_resnet101(video_path)  # 提取特征并更新全局变量
    result_label.config(text=f"resnet101特征提取完成，特征长度：{len(video_features)}")  # 更新标签内容显示特征长度
    print(video_path)
    # print(video_features)

def extract_and_display_features2_resnet101(video_path): #更新全局变量并显示 调用模型
    global video_features
    video_features = extract_features2_CNN_resnet101(video_path)  # 提取特征并更新全局变量
    result_label.config(text=f"resnet101特征提取完成，特征长度：{len(video_features)}")  # 更新标签内容显示特征长度
    print(video_path)
    # print(video_features)

def extract_and_display_features3_resnet101(video_path): #更新全局变量并显示 调用模型
    global video_features
    video_features = extract_features3_CNN_resnet101(video_path)  # 提取特征并更新全局变量
    result_label.config(text=f"resnet101特征提取完成，特征长度：{len(video_features)}")  # 更新标签内容显示特征长度
    print(video_path)
    # print(video_features)

def extract_features_CNN_resnet101(video_path):  # CNN_resnet101模型提取特征
    # 使用更新的权重选择方式初始化ResNet模型
    weights = ResNet101_Weights.DEFAULT  # ResNet101默认权重配置
    model = models.resnet101(weights=weights)  # 加载预训练的ResNet模型
    model.eval()  # 设置为评估模式

    if torch.cuda.is_available():
        model = model.to('cuda')  # 判断当前环境是否可以使用GPU

    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),  # 缩放为256*256
        transforms.CenterCrop(224),  # 中心剪裁为224*224
        transforms.ToTensor(),  # 将裁剪后的图像转换为张量格式，以便模型能够处理
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])  # 对张量进行归一化处理，使用指定的均值和标准差

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while True:
        ret, frame = cap.read()  # ret 成功显示成功与否  frame 读取到的帧
        if not ret:
            break

        # 应用预处理并添加批次维度
        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            output = model(input_batch)

        # 将特征转换为一维列表并存储
        features.append(output.cpu().numpy().flatten())

        # 更新进度条
        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    cap.release()
    return features

def dimension_reduction_pca_resnet101(video_features): #实现了利用主成分分析（PCA）来降低视频特征的维度，并且使用散点图来可视化降维后的结果
    # 假设 features 是你所有帧的特征列表，每个元素是一个特征向量

    features = np.array(video_features) #将视频特征列表转为数组
    # 应用PCA
    pca = PCA(n_components=2)  # 降至2维
    reduced_features = pca.fit_transform(video_features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

# 第一层
def extract_features1_CNN_resnet101(video_path):
    weights = ResNet101_Weights.DEFAULT
    model = models.resnet101(weights=weights)
    model.eval()

    if torch.cuda.is_available():
        model = model.to('cuda')

    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    features1 = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # 定义一个钩子函数来捕获第一层的输出
    layer_features1 = None

    def hook(module, input, output):
        nonlocal layer_features1
        layer_features1 = output

    model.conv1.register_forward_hook(hook)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            model(input_batch)

        # 将第一层特征转换为一维列表并存储
        if layer_features1 is not None:
            features1.append(layer_features1.cpu().numpy().flatten())

        # 更新进度条
        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    cap.release()
    return features1

def dimension_reduction_pca1_resnet101(video_features): #实现了利用主成分分析（PCA）来降低视频特征的维度，并且使用散点图来可视化降维后的结果
    # 假设 features 是你所有帧的特征列表，每个元素是一个特征向量

    features1 = np.array(video_features) #将视频特征列表转为数组
    # 应用PCA
    pca = PCA(n_components=2)  # 降至2维
    reduced_features = pca.fit_transform(video_features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

# 第二层
def extract_features2_CNN_resnet101(video_path):
    weights = ResNet101_Weights.DEFAULT
    model = models.resnet101(weights=weights)
    model.eval()

    if torch.cuda.is_available():
        model = model.to('cuda')

    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    features2 = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # 定义一个钩子函数来捕获第一层的输出
    layer_features2 = None

    def hook(module, input, output):
        nonlocal layer_features2
        layer_features2 = output

    model.layer1.register_forward_hook(hook)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            model(input_batch)

        # 将第一层特征转换为一维列表并存储
        if layer_features2 is not None:
            features2.append(layer_features2.cpu().numpy().flatten())

        # 更新进度条
        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    cap.release()
    return features2

def dimension_reduction_pca2_resnet101(video_features): #实现了利用主成分分析（PCA）来降低视频特征的维度，并且使用散点图来可视化降维后的结果
    # 假设 features 是你所有帧的特征列表，每个元素是一个特征向量

    features2 = np.array(video_features) #将视频特征列表转为数组
    # 应用PCA
    pca = PCA(n_components=2)  # 降至2维
    reduced_features = pca.fit_transform(video_features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

# 第三层
def extract_features3_CNN_resnet101(video_path):
    weights = ResNet101_Weights.DEFAULT
    model = models.resnet101(weights=weights)
    model.eval()

    if torch.cuda.is_available():
        model = model.to('cuda')

    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    features3 = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # 定义一个钩子函数来捕获第一层的输出
    layer_features3 = None

    def hook(module, input, output):
        nonlocal layer_features3
        layer_features3 = output

    model.layer2.register_forward_hook(hook)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            model(input_batch)

        # 将第一层特征转换为一维列表并存储
        if layer_features3 is not None:
            features3.append(layer_features3.cpu().numpy().flatten())

        # 更新进度条
        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    cap.release()
    return features3

def dimension_reduction_pca3_resnet101(video_features): #实现了利用主成分分析（PCA）来降低视频特征的维度，并且使用散点图来可视化降维后的结果
    # 假设 features 是你所有帧的特征列表，每个元素是一个特征向量

    features3 = np.array(video_features) #将视频特征列表转为数组
    # 应用PCA
    pca = PCA(n_components=2)  # 降至2维
    reduced_features = pca.fit_transform(video_features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()


def feacture_change_dynamics_resnet101(video_features):
    # 准备画布
    fig, ax = plt.subplots()
    frame_features = video_features[0].reshape((25, 40))
    img = ax.imshow(frame_features, cmap='viridis')

    def update(frame_number):
        frame_features = video_features[frame_number].reshape((25, 40))
        img.set_data(frame_features)
        ax.set_title(f"Frame {frame_number + 1}")
        return img,

    # 创建动画
    # ani = FuncAnimation(fig, update, frames=278, interval=50)  # 每帧间隔50毫秒
    ani = FuncAnimation(fig, update, frames=len(video_features), interval=50)
    plt.show()


#resnet152
def use_extract_features_CNN_resnet152():  # 选择视频
    # 打开文件选择对话框让用户选择视频文件
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:  # 确保用户选择了文件
        threading.Thread(target=extract_and_display_features_resnet152, args=(video_path,)).start()

def use_extract_features1_CNN_resnet152():  # 选择视频
    # 打开文件选择对话框让用户选择视频文件
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:  # 确保用户选择了文件
        threading.Thread(target=extract_and_display_features1_resnet152, args=(video_path,)).start()

def use_extract_features2_CNN_resnet152():  # 选择视频
    # 打开文件选择对话框让用户选择视频文件
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:  # 确保用户选择了文件
        threading.Thread(target=extract_and_display_features2_resnet152, args=(video_path,)).start()

def use_extract_features3_CNN_resnet152():  # 选择视频
    # 打开文件选择对话框让用户选择视频文件
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:  # 确保用户选择了文件
        threading.Thread(target=extract_and_display_features3_resnet152, args=(video_path,)).start()

def extract_and_display_features_resnet152(video_path): #更新全局变量并显示 调用模型
    global video_features
    video_features = extract_features_CNN_resnet152(video_path)  # 提取特征并更新全局变量
    result_label.config(text=f"resnet152特征提取完成，特征长度：{len(video_features)}")  # 更新标签内容显示特征长度
    print(video_path)
    # print(video_features)

def extract_and_display_features1_resnet152(video_path): #更新全局变量并显示 调用模型
    global video_features
    video_features = extract_features1_CNN_resnet152(video_path)  # 提取特征并更新全局变量
    result_label.config(text=f"resnet152特征提取完成，特征长度：{len(video_features)}")  # 更新标签内容显示特征长度
    print(video_path)
    # print(video_features)

def extract_and_display_features2_resnet152(video_path): #更新全局变量并显示 调用模型
    global video_features
    video_features = extract_features2_CNN_resnet152(video_path)  # 提取特征并更新全局变量
    result_label.config(text=f"resnet152特征提取完成，特征长度：{len(video_features)}")  # 更新标签内容显示特征长度
    print(video_path)
    # print(video_features)

def extract_and_display_features3_resnet152(video_path): #更新全局变量并显示 调用模型
    global video_features
    video_features = extract_features3_CNN_resnet152(video_path)  # 提取特征并更新全局变量
    result_label.config(text=f"resnet152特征提取完成，特征长度：{len(video_features)}")  # 更新标签内容显示特征长度
    print(video_path)
    # print(video_features)

def extract_features_CNN_resnet152(video_path):  # CNN_resnet152模型提取特征
    # 使用更新的权重选择方式初始化ResNet模型
    weights = ResNet152_Weights.DEFAULT  # ResNet152默认权重配置
    model = models.resnet152(weights=weights)  # 加载预训练的ResNet模型
    model.eval()  # 设置为评估模式

    if torch.cuda.is_available():
        model = model.to('cuda')  # 判断当前环境是否可以使用GPU

    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),  # 缩放为256*256
        transforms.CenterCrop(224),  # 中心剪裁为224*224
        transforms.ToTensor(),  # 将裁剪后的图像转换为张量格式，以便模型能够处理
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])  # 对张量进行归一化处理，使用指定的均值和标准差

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while True:
        ret, frame = cap.read()  # ret 成功显示成功与否  frame 读取到的帧
        if not ret:
            break

        # 应用预处理并添加批次维度
        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            output = model(input_batch)

        # 将特征转换为一维列表并存储
        features.append(output.cpu().numpy().flatten())

        # 更新进度条
        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    cap.release()
    return features

def dimension_reduction_pca_resnet152(video_features): #实现了利用主成分分析（PCA）来降低视频特征的维度，并且使用散点图来可视化降维后的结果
    # 假设 features 是你所有帧的特征列表，每个元素是一个特征向量

    features = np.array(video_features) #将视频特征列表转为数组
    # 应用PCA
    pca = PCA(n_components=2)  # 降至2维
    reduced_features = pca.fit_transform(video_features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

# 第一层
def extract_features1_CNN_resnet152(video_path):
    weights = ResNet152_Weights.DEFAULT
    model = models.resnet152(weights=weights)
    model.eval()

    if torch.cuda.is_available():
        model = model.to('cuda')

    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    features1 = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # 定义一个钩子函数来捕获第一层的输出
    layer_features1 = None

    def hook(module, input, output):
        nonlocal layer_features1
        layer_features1 = output

    model.conv1.register_forward_hook(hook)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            model(input_batch)

        # 将第一层特征转换为一维列表并存储
        if layer_features1 is not None:
            features1.append(layer_features1.cpu().numpy().flatten())

        # 更新进度条
        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    cap.release()
    return features1

def dimension_reduction_pca1_resnet152(video_features): #实现了利用主成分分析（PCA）来降低视频特征的维度，并且使用散点图来可视化降维后的结果
    # 假设 features 是你所有帧的特征列表，每个元素是一个特征向量

    features1 = np.array(video_features) #将视频特征列表转为数组
    # 应用PCA
    pca = PCA(n_components=2)  # 降至2维
    reduced_features = pca.fit_transform(video_features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

# 第二层
def extract_features2_CNN_resnet152(video_path):
    weights = ResNet152_Weights.DEFAULT
    model = models.resnet152(weights=weights)
    model.eval()

    if torch.cuda.is_available():
        model = model.to('cuda')

    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    features2 = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # 定义一个钩子函数来捕获第一层的输出
    layer_features2 = None

    def hook(module, input, output):
        nonlocal layer_features2
        layer_features2 = output

    model.layer1.register_forward_hook(hook)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            model(input_batch)

        # 将第二层特征转换为一维列表并存储
        if layer_features2 is not None:
            features2.append(layer_features2.cpu().numpy().flatten())

        # 更新进度条
        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    cap.release()
    return features2

def dimension_reduction_pca2_resnet152(video_features): #实现了利用主成分分析（PCA）来降低视频特征的维度，并且使用散点图来可视化降维后的结果
    # 假设 features 是你所有帧的特征列表，每个元素是一个特征向量

    features2 = np.array(video_features) #将视频特征列表转为数组
    # 应用PCA
    pca = PCA(n_components=2)  # 降至2维
    reduced_features = pca.fit_transform(video_features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

# 第三层
def extract_features3_CNN_resnet152(video_path):
    weights = ResNet152_Weights.DEFAULT
    model = models.resnet152(weights=weights)
    model.eval()

    if torch.cuda.is_available():
        model = model.to('cuda')

    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    features3 = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # 定义一个钩子函数来捕获第一层的输出
    layer_features3 = None

    def hook(module, input, output):
        nonlocal layer_features3
        layer_features3 = output

    model.layer2.register_forward_hook(hook)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            model(input_batch)

        # 将第一层特征转换为一维列表并存储
        if layer_features3 is not None:
            features3.append(layer_features3.cpu().numpy().flatten())

        # 更新进度条
        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    cap.release()
    return features3

def dimension_reduction_pca3_resnet152(video_features): #实现了利用主成分分析（PCA）来降低视频特征的维度，并且使用散点图来可视化降维后的结果
    # 假设 features 是你所有帧的特征列表，每个元素是一个特征向量

    features3 = np.array(video_features) #将视频特征列表转为数组
    # 应用PCA
    pca = PCA(n_components=2)  # 降至2维
    reduced_features = pca.fit_transform(video_features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()


def feacture_change_dynamics_resnet152(video_features):
    # 准备画布
    fig, ax = plt.subplots()
    frame_features = video_features[0].reshape((25, 40))
    img = ax.imshow(frame_features, cmap='viridis')

    def update(frame_number):
        frame_features = video_features[frame_number].reshape((25, 40))
        img.set_data(frame_features)
        ax.set_title(f"Frame {frame_number + 1}")
        return img,

    # 创建动画
    # ani = FuncAnimation(fig, update, frames=278, interval=50)  # 每帧间隔50毫秒
    ani = FuncAnimation(fig, update, frames=len(video_features), interval=50)
    plt.show()


#resnet200
def use_extract_features_CNN_resnet200():  # 选择视频
    # 打开文件选择对话框让用户选择视频文件
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:  # 确保用户选择了文件
        threading.Thread(target=extract_and_display_features_resnet200, args=(video_path,)).start()

def use_extract_features1_CNN_resnet200():  # 选择视频
    # 打开文件选择对话框让用户选择视频文件
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:  # 确保用户选择了文件
        threading.Thread(target=extract_and_display_features1_resnet200, args=(video_path,)).start()

def use_extract_features2_CNN_resnet200():  # 选择视频
    # 打开文件选择对话框让用户选择视频文件
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:  # 确保用户选择了文件
        threading.Thread(target=extract_and_display_features2_resnet200, args=(video_path,)).start()

def use_extract_features3_CNN_resnet200():  # 选择视频
    # 打开文件选择对话框让用户选择视频文件
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:  # 确保用户选择了文件
        threading.Thread(target=extract_and_display_features3_resnet200, args=(video_path,)).start()

def extract_and_display_features_resnet200(video_path): #更新全局变量并显示 调用模型
    global video_features
    video_features = extract_features_CNN_resnet200(video_path)  # 提取特征并更新全局变量
    result_label.config(text=f"resnet200特征提取完成，特征长度：{len(video_features)}")  # 更新标签内容显示特征长度
    print(video_path)
    # print(video_features)

def extract_and_display_features1_resnet200(video_path): #更新全局变量并显示 调用模型
    global video_features
    video_features = extract_features1_CNN_resnet200(video_path)  # 提取特征并更新全局变量
    result_label.config(text=f"resnet200特征提取完成，特征长度：{len(video_features)}")  # 更新标签内容显示特征长度
    print(video_path)
    # print(video_features)

def extract_and_display_features2_resnet200(video_path): #更新全局变量并显示 调用模型
    global video_features
    video_features = extract_features2_CNN_resnet200(video_path)  # 提取特征并更新全局变量
    result_label.config(text=f"resnet200特征提取完成，特征长度：{len(video_features)}")  # 更新标签内容显示特征长度
    print(video_path)
    # print(video_features)

def extract_and_display_features3_resnet200(video_path): #更新全局变量并显示 调用模型
    global video_features
    video_features = extract_features3_CNN_resnet200(video_path)  # 提取特征并更新全局变量
    result_label.config(text=f"resnet200特征提取完成，特征长度：{len(video_features)}")  # 更新标签内容显示特征长度
    print(video_path)
    # print(video_features)

def extract_features_CNN_resnet200(video_path):  # CNN_resnet200模型提取特征
    # 使用更新的权重选择方式初始化ResNet模型
    weights = ResNet200_Weights.DEFAULT  # ResNet200默认权重配置
    model = models.resnet200(weights=weights)  # 加载预训练的ResNet模型
    model.eval()  # 设置为评估模式

    if torch.cuda.is_available():
        model = model.to('cuda')  # 判断当前环境是否可以使用GPU

    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),  # 缩放为256*256
        transforms.CenterCrop(224),  # 中心剪裁为224*224
        transforms.ToTensor(),  # 将裁剪后的图像转换为张量格式，以便模型能够处理
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])  # 对张量进行归一化处理，使用指定的均值和标准差

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while True:
        ret, frame = cap.read()  # ret 成功显示成功与否  frame 读取到的帧
        if not ret:
            break

        # 应用预处理并添加批次维度
        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            output = model(input_batch)

        # 将特征转换为一维列表并存储
        features.append(output.cpu().numpy().flatten())

        # 更新进度条
        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    cap.release()
    return features

def dimension_reduction_pca_resnet200(video_features): #实现了利用主成分分析（PCA）来降低视频特征的维度，并且使用散点图来可视化降维后的结果
    # 假设 features 是你所有帧的特征列表，每个元素是一个特征向量

    features = np.array(video_features) #将视频特征列表转为数组
    # 应用PCA
    pca = PCA(n_components=2)  # 降至2维
    reduced_features = pca.fit_transform(video_features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

# 第一层
def extract_features1_CNN_resnet200(video_path):
    weights = ResNet200_Weights.DEFAULT
    model = models.resnet200(weights=weights)
    model.eval()

    if torch.cuda.is_available():
        model = model.to('cuda')

    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    features1 = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # 定义一个钩子函数来捕获第一层的输出
    layer_features1 = None

    def hook(module, input, output):
        nonlocal layer_features1
        layer_features1 = output

    model.conv1.register_forward_hook(hook)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            model(input_batch)

        # 将第一层特征转换为一维列表并存储
        if layer_features1 is not None:
            features1.append(layer_features1.cpu().numpy().flatten())

        # 更新进度条
        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    cap.release()
    return features1

def dimension_reduction_pca1_resnet200(video_features): #实现了利用主成分分析（PCA）来降低视频特征的维度，并且使用散点图来可视化降维后的结果
    # 假设 features 是你所有帧的特征列表，每个元素是一个特征向量

    features1 = np.array(video_features) #将视频特征列表转为数组
    # 应用PCA
    pca = PCA(n_components=2)  # 降至2维
    reduced_features = pca.fit_transform(video_features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

# 第二层
def extract_features2_CNN_resnet200(video_path):
    weights = ResNet200_Weights.DEFAULT
    model = models.resnet200(weights=weights)
    model.eval()

    if torch.cuda.is_available():
        model = model.to('cuda')

    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    features2 = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # 定义一个钩子函数来捕获第一层的输出
    layer_features2 = None

    def hook(module, input, output):
        nonlocal layer_features2
        layer_features2 = output

    model.layer1.register_forward_hook(hook)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            model(input_batch)

        # 将第一层特征转换为一维列表并存储
        if layer_features2 is not None:
            features2.append(layer_features2.cpu().numpy().flatten())

        # 更新进度条
        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    cap.release()
    return features2

def dimension_reduction_pca2_resnet200(video_features): #实现了利用主成分分析（PCA）来降低视频特征的维度，并且使用散点图来可视化降维后的结果
    # 假设 features 是你所有帧的特征列表，每个元素是一个特征向量

    features2 = np.array(video_features) #将视频特征列表转为数组
    # 应用PCA
    pca = PCA(n_components=2)  # 降至2维
    reduced_features = pca.fit_transform(video_features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

# 第三层
def extract_features3_CNN_resnet200(video_path):
    weights = ResNet200_Weights.DEFAULT
    model = models.resnet200(weights=weights)
    model.eval()

    if torch.cuda.is_available():
        model = model.to('cuda')

    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    features3 = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # 定义一个钩子函数来捕获第一层的输出
    layer_features3 = None

    def hook(module, input, output):
        nonlocal layer_features3
        layer_features3 = output

    model.layer2.register_forward_hook(hook)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            model(input_batch)

        # 将第一层特征转换为一维列表并存储
        if layer_features3 is not None:
            features3.append(layer_features3.cpu().numpy().flatten())

        # 更新进度条
        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    cap.release()
    return features3

def dimension_reduction_pca3_resnet200(video_features): #实现了利用主成分分析（PCA）来降低视频特征的维度，并且使用散点图来可视化降维后的结果
    # 假设 features 是你所有帧的特征列表，每个元素是一个特征向量

    features3 = np.array(video_features) #将视频特征列表转为数组
    # 应用PCA
    pca = PCA(n_components=2)  # 降至2维
    reduced_features = pca.fit_transform(video_features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()


def feacture_change_dynamics_resnet200(video_features):
    # 准备画布
    fig, ax = plt.subplots()
    frame_features = video_features[0].reshape((25, 40))
    img = ax.imshow(frame_features, cmap='viridis')

    def update(frame_number):
        frame_features = video_features[frame_number].reshape((25, 40))
        img.set_data(frame_features)
        ax.set_title(f"Frame {frame_number + 1}")
        return img,

    # 创建动画
    # ani = FuncAnimation(fig, update, frames=278, interval=50)  # 每帧间隔50毫秒
    ani = FuncAnimation(fig, update, frames=len(video_features), interval=50)
    plt.show()


#VGG16
#全局
def use_extract_features_VGG16():  # 选择文件
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:  # 确保用户选择了文件
        threading.Thread(target=extract_and_display_features_VGG16, args=(video_path,)).start()


def extract_and_display_features_VGG16(video_path):  # 更新全局变量并显示 调用模型
    global video_features
    video_features = extract_features_VGG16(video_path)  # 提取特征并更新全局变量
    result_label.config(text=f"VGG16特征提取完成，特征长度：{len(video_features)}")  # 更新标签内容显示特征长度
    print(video_path)
    # print(video_features)



def extract_features_VGG16(video_path):  # VGG16模型提取特征
    model = models.vgg16(pretrained=True)  # 加载预训练的VGG16模型
    model.eval()  # 设置为评估模式

    if torch.cuda.is_available():
        model = model.to('cuda')

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # 将裁剪后的图像转换为张量格式，以便模型能够处理
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])  # 对张量进行归一化处理，使用指定的均值和标准差

    cap = cv2.VideoCapture(video_path)
    features = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while True:
        ret, frame = cap.read()  # ret 成功显示成功与否 frame 读取到的帧
        if not ret:
            break

        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            output = model.features(input_batch)

        features.append(output.cpu().numpy().flatten())

        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    cap.release()
    return features



def dimension_reduction_pca_VGG16(video_features):  # 利用PCA降低特征维度
    features = np.array(video_features)  # 将视频特征列表转为数组
    pca = PCA(n_components=2)  # 降至2维
    reduced_features = pca.fit_transform(video_features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

def dimension_reduction_pca_first_layer(first_layer_features):
    features = np.array(first_layer_features)
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7, label='First Layer')
    plt.title('PCA Result for First Layer')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.legend()
    plt.show()

def dimension_reduction_pca_middle_layer(second_layer_features):
    features = np.array(second_layer_features)
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7, label='Second Layer')
    plt.title('PCA Result for Second Layer')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.legend()
    plt.show()

def dimension_reduction_pca_last_layer(third_layer_features):
    features = np.array(third_layer_features)
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7, label='Third Layer')
    plt.title('PCA Result for Second Layer')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.legend()
    plt.show()


def feature_change_dynamics_VGG16(video_features):
    fig, ax = plt.subplots()
    frame_features = video_features[0].reshape((7, 7, 512))
    img = ax.imshow(frame_features[:, :, 0], cmap='viridis')  # 仅显示第一通道

    def update(frame_number):
        frame_features = video_features[frame_number].reshape((7, 7, 512))
        img.set_data(frame_features[:, :, 0])  # 更新第一通道数据
        ax.set_title(f"Frame {frame_number + 1}")
        return img,

    ani = FuncAnimation(fig, update, frames=len(video_features), interval=50)  # 每帧间隔50毫秒
    plt.show()






#inception
def use_extract_features_inception():
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:
        threading.Thread(target=extract_and_display_features_inception, args=(video_path,)).start()

def extract_and_display_features_inception(video_path):
    global video_features
    video_features = extract_features_inception(video_path)
    result_label.config(text=f"Inception特征提取完成，特征长度：{len(video_features)}")
    print(video_path)
    # print(video_features)


def extract_features_inception(video_path):
    model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.eval()

    if torch.cuda.is_available():
        model = model.to('cuda')

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(video_path)
    features = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            output = model(input_batch)

        features.append(output.cpu().numpy().flatten())

        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    cap.release()
    return features

def visualize_features_inception(features):
    features = np.array(features)
    plt.figure(figsize=(10, 6))
    plt.imshow(features, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Inception Video Features')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Frame')
    plt.show()



def dimension_reduction_pca_inception(video_features):  # 利用PCA降低特征维度
    features = np.array(video_features)  # 将视频特征列表转为数组
    pca = PCA(n_components=2)  # 降至2维
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()


def feature_change_dynamics_inception(video_features):
    fig, ax = plt.subplots()

    # 假设 Inception V3 的特征向量为 1000 维
    frame_features = video_features[0].reshape( (25, 40))  # 这里的形状可以根据实际情况调整
    img = ax.imshow(frame_features, cmap='viridis')

    def update(frame_number):
        frame_features = video_features[frame_number].reshape( (25, 40))  # 调整为合适的形状
        img.set_data(frame_features)
        ax.set_title(f"Frame {frame_number + 1}")
        return img,

    ani = FuncAnimation(fig, update, frames=len(video_features), interval=50)  # 每帧间隔50毫秒
    plt.show()





from torchvision.models import DenseNet121_Weights
# denseNet# 初始化DenseNet121模型并加载预训练权重
weights = DenseNet121_Weights.DEFAULT
model_densenet121 = models.densenet121(weights=weights)
model_densenet121.eval()

if torch.cuda.is_available():
    model_densenet121 = model_densenet121.to('cuda')

# 定义预处理操作
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def use_extract_features_CNN_densenet121():
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:
        print(f"Selected video path: {video_path}")  # 打印选择的视频路径以调试
        threading.Thread(target=extract_and_display_features_densenet121, args=(video_path,)).start()


def extract_and_display_features_densenet121(video_path):
    global video_features
    video_features = extract_features_CNN_densenet121(video_path)
    result_label.config(text=f"DenseNet121特征提取完成，特征长度：{len(video_features)}")
    # if video_features:
        # dimension_reduction_pca_densenet121(video_features)
        # feature_change_dynamics_densenet121(video_features)

def extract_features_CNN_densenet121(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            output = model_densenet121(input_batch)

        features.append(output.cpu().numpy().flatten())

        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    cap.release()
    return features

def dimension_reduction_pca_densenet121(video_features):
    features = np.array(video_features)
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

def feature_change_dynamics_densenet121(video_features):
    fig, ax = plt.subplots()
    frame_features = video_features[0].reshape((25, 40))  # 适应DenseNet121的特征
    img = ax.imshow(frame_features, cmap='viridis')

    def update(frame_number):
        frame_features = video_features[frame_number].reshape((25, 40))
        img.set_data(frame_features)
        ax.set_title(f"Frame {frame_number + 1}")
        return img,

    ani = FuncAnimation(fig, update, frames=len(video_features), interval=50)
    plt.show()


def visualize_features_densenet(features):
    features = np.array(features)
    plt.figure(figsize=(10, 6))
    plt.imshow(features, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Inception Video Features')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Frame')
    plt.show()




#加进度条
# 视频预处理函数
def video_preprocessing_C3D(video_path, frame_count=16, resize=(112, 112)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    current_frame = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // frame_count, 1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, resize)
                frames.append(frame)
                if len(frames) == frame_count:
                    break
            current_frame += 1
            progress = (current_frame / total_frames) * 100
            progress_bar['value'] = progress
            root.update_idletasks()
    finally:
        cap.release()

    # 确保进度条到达100%
    progress_bar['value'] = 100
    root.update_idletasks()

    frames = np.array(frames, dtype=np.float32)
    frames = np.transpose(frames, (3, 0, 1, 2))
    frames = np.expand_dims(frames, axis=0)
    return torch.tensor(frames)



# 模型定义
class C3DFeatureExtractor(nn.Module):
    def __init__(self):
        super(C3DFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.relu = nn.ReLU()
        # 添加更多层根据需要

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu(x)
        # 添加更多层的前向传递
        return x

def load_video_and_extract_3dfeatures():
    global video_features
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:
        threading.Thread(target=extract_and_display_3dfeatures, args=(video_path,)).start()

def extract_and_display_3dfeatures(video_path):
    global video_features
    video_features = extract_3dfeatures(video_path)
    result_label.config(text="C3D特征提取完成")

def extract_3dfeatures(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = C3DFeatureExtractor().to(device)
    model.eval()

    video_tensor = video_preprocessing_C3D(video_path).to(device)

    with torch.no_grad():
        features = model(video_tensor)

    print(features.shape)
    return features


# # 定义更复杂的C3D模型
# class C3DFeatureExtractor(nn.Module):
#     def __init__(self):
#         super(C3DFeatureExtractor, self).__init__()
#
#         self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1)
#         self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
#
#         self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
#         self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#
#         self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
#         self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=1)
#         self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#
#         self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=1)
#         self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)
#         self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#
#         self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)
#         self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)
#         self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.pool1(x)
#
#         x = self.relu(self.conv2(x))
#         x = self.pool2(x)
#
#         x = self.relu(self.conv3a(x))
#         x = self.relu(self.conv3b(x))
#         x = self.pool3(x)
#
#         x = self.relu(self.conv4a(x))
#         x = self.relu(self.conv4b(x))
#         x = self.pool4(x)
#
#         x = self.relu(self.conv5a(x))
#         x = self.relu(self.conv5b(x))
#         x = self.pool5(x)
#
#         return x
#
#
# # 视频预处理函数
# def video_preprocessing_C3D(video_path):
#     frames = []
#     cap = cv2.VideoCapture(video_path)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     current_frame = 0
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.resize(frame, (112, 112))
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame = frame / 255.0
#         frames.append(frame)
#
#         current_frame += 1
#         progress = (current_frame / frame_count) * 100
#         progress_bar['value'] = progress
#         root.update_idletasks()
#
#     cap.release()
#
#     frames = torch.tensor(frames).permute(3, 0, 1, 2).unsqueeze(0).float()  # [batch, channel, depth, height, width]
#     return frames
#
#
# # 特征提取函数
# def extract_3dfeatures(video_path):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = C3DFeatureExtractor().to(device)
#     model.eval()
#
#     video_tensor = video_preprocessing_C3D(video_path).to(device)
#
#     with torch.no_grad():
#         features = model(video_tensor)
#
#     return features
#
#
# # 文件选择并提取特征
# def load_video_and_extract_3dfeatures():
#     global video_features
#     video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
#     if video_path:
#         video_features = extract_3dfeatures(video_path)
#         result_label.config(text="C3D特征提取完成")
#         print(video_features)


def visualize_feature_maps(features, num_maps=36):    #视频每一帧的特征图
    features = features.squeeze(0)  # 假设features是形状为[1, channels, depth, height, width]的张量  去掉批量维度（将代表第几帧的标签去掉）
    if features.shape[0] < num_maps:  #如果特征图超过实际可用的数量 抛出错误
        raise ValueError(f"Requested {num_maps} feature maps, but only {features.shape[0]} are available.")

    fig, axes = plt.subplots(5, 5, figsize=(15, 15))  # 创建一个5×5的子图网格
    axes = axes.flatten()  # 将二维的axes数组转化为一维，便于索引

    for i, ax in enumerate(axes[:num_maps]):  # 只迭代前 num_maps 个元素
        # 选择第i个特征图，取第一个时间点的数据
        feature_map = features[i, 0].cpu().detach().numpy()
        ax.imshow(feature_map, cmap='viridis')
        ax.axis('off')
        ax.set_title(f'Feature {i+1}', fontsize=10)  # 可选：为每个特征图添加标题

    # 关闭剩余的轴（如果特征图数量少于36）
    for ax in axes[num_maps:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_feature_histograms(features, num_bins=30):  #直方图
    features = features.squeeze(0)  # 假设 features 是形状为 [1, channels, depth, height, width] 的张量  去掉批量维度
    plt.figure(figsize=(10, 5))
    plt.hist(features.cpu().numpy().ravel(), bins=num_bins, color='blue', alpha=0.7)
    plt.title("Feature Histogram")
    plt.xlabel("Feature value")
    plt.ylabel("Frequency")
    plt.show()





#visualize_heatmaps 函数旨在将从神经网络（如卷积神经网络）提取的特征图以热图（heatmap）的形式可视化。
# 这种可视化有助于理解网络中各层如何响应不同的输入特征，特别是在处理图像或体积数据时

def visualize_heatmaps(features, num_maps=36):  # 更新 num_maps 为 36
    features = features.squeeze(0)
    if features.shape[0] < num_maps:
        print(f"Warning: Requested {num_maps} feature maps, but only {features.shape[0]} are available.")
        num_maps = features.shape[0]  # 如果不足36张，则调整为实际数量

    fig, axes = plt.subplots(6, 6, figsize=(15, 15))  # 创建6x6的子图网格
    axes = axes.flatten()  # 将 axes 从二维数组转换为一维数组以便遍历

    for i, ax in enumerate(axes):  #同时获取子图的索引 i 和对应的子图对象 ax
        if i < num_maps:  # 仅处理有数据的部分
            feature_map = features[i, 0].cpu().detach().numpy()  # 取第 i 个特征图的第一个时间点
            if feature_map.ndim == 2:
                heatmap = feature_map
            elif feature_map.ndim == 3:
                heatmap = np.sum(feature_map, axis=0)  # 对深度维度求和，生成二维热图
            else:
                print(f"Skipping feature map {i} due to unexpected dimensions: {feature_map.ndim}")
                continue

            if heatmap.ndim != 2:
                print(f"Invalid heatmap dimension for feature map {i}: {heatmap.ndim}")
                continue

            ax.imshow(heatmap, cmap='hot', interpolation='nearest')
            ax.axis('off')
        else:
            ax.axis('off')  # 对于多余的轴，关闭它们

    plt.tight_layout()
    plt.show()



def animate_feature_maps(features):   #特征图动态序列显示
    features = features.cpu().numpy()
    fig, ax = plt.subplots()
    ims = []

    for i in range(features.shape[2]):  # 遍历时间维度
        im = ax.imshow(features[0, 0, i], cmap='viridis', animated=True)
        if i == 0:
            ax.imshow(features[0, 0, i], cmap='viridis')  # 显示第一帧以设置颜色映射和格式
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    plt.show()


def animate_heatmaps(features):   #热力图动态序列显示
    features = features.cpu().numpy()
    fig, ax = plt.subplots()
    ims = []

    for i in range(features.shape[2]):  # 遍历时间维度
        heatmap = np.sum(features[0, :, i, :, :], axis=0)  # 对深度维度求和
        im = ax.imshow(heatmap, cmap='hot', animated=True)
        if i == 0:
            ax.imshow(heatmap, cmap='hot')  # 显示第一帧以设置颜色映射和格式
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    plt.show()





class VideoFeatureExtractorMultiLayerLSTM(nn.Module):
    def __init__(self):
        super(VideoFeatureExtractorMultiLayerLSTM, self).__init__()
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 512)

        self.lstm1 = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_out = c_out.view(batch_size, timesteps, -1)

        lstm_out1, _ = self.lstm1(r_out)
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out3, _ = self.lstm3(lstm_out2)

        return lstm_out1, lstm_out2, lstm_out3
def video_preprocessingls(video_path, frame_count=16, resize=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    frame_interval = max(total_frames // frame_count, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(frame)
            frames.append(frame)

            if len(frames) == frame_count:
                break

        current_frame += 1
        progress = (current_frame / total_frames) * 100
        progress_bar['value'] = progress
        root.update_idletasks()

    # 确保进度条到达100%
    progress_bar['value'] = 100
    root.update_idletasks()

    cap.release()
    frames = torch.stack(frames)
    frames = frames.unsqueeze(0)
    return frames

# def video_preprocessingls(video_path, frame_count=16, resize=(224, 224)):
#     cap = cv2.VideoCapture(video_path)
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize(resize),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     frames = []
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     current_frame = 0
#     frame_interval = max(total_frames // frame_count, 1)
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if current_frame % frame_interval == 0:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = transform(frame)
#             frames.append(frame)
#
#             if len(frames) == frame_count:
#                 break
#
#         current_frame += 1
#         progress = (current_frame / total_frames) * 100
#         progress_bar['value'] = progress
#         root.update_idletasks()
#
#     cap.release()
#     frames = torch.stack(frames)
#     frames = frames.unsqueeze(0)
#     return frames

def load_video_and_extract_multilayer_lstm_features():
    global video_features
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:
        threading.Thread(target=extract_and_display_multilayer_lstm_features, args=(video_path,)).start()

def extract_and_display_multilayer_lstm_features(video_path):
    global video_features
    video_features = extract_multilayer_lstm1_features(video_path)
    result_label.config(text="Multi-Layer LSTM 特征提取完成")

def extract_multilayer_lstm1_features(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoFeatureExtractorMultiLayerLSTM().to(device)
    model.eval()
    frames = video_preprocessingls(video_path).to(device)
    with torch.no_grad():
        features1, features2, features3 = model(frames)

    print(features1.shape, features2.shape, features3.shape)
    return features1



def extract_multilayer_lstm2_features(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoFeatureExtractorMultiLayerLSTM().to(device)
    model.eval()
    frames = video_preprocessingls(video_path)
    frames = frames.to(device)
    with torch.no_grad():
        features1, features2, features3 = model(frames)

    print(features1.shape, features2.shape, features3.shape)
    return features2

def extract_multilayer_lstm3_features(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoFeatureExtractorMultiLayerLSTM().to(device)
    model.eval()
    frames = video_preprocessingls(video_path)
    frames = frames.to(device)
    with torch.no_grad():
        features1, features2, features3 = model(frames)

    print(features1.shape, features2.shape, features3.shape)
    return features3


def load_video_and_extract_multilayer_lstm1_features():
    global video_features
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:
        video_features = extract_multilayer_lstm1_features(video_path)
        result_label.config(text="第一层LSTM特征提取完成")


def load_video_and_extract_multilayer_lstm2_features():
    global video_features
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:
        video_features = extract_multilayer_lstm2_features(video_path)
        result_label.config(text="第二层LSTM特征提取完成")

def load_video_and_extract_multilayer_lstm3_features():
    global video_features
    video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if video_path:
        video_features = extract_multilayer_lstm3_features(video_path)
        result_label.config(text="第三层LSTM特征提取完成")


def visualize_lsfeaturesls(features):
    # 重新整理特征的形状以匹配 t-SNE 的输入需求
    features = features.reshape(-1, features.shape[-1])  # 确保特征是二维的

    # 确保张量在CPU上并转换为numpy数组
    if torch.is_tensor(features):
        features = features.cpu().detach().numpy()

    # 实际样本数量
    n_samples = features.shape[0]

    # 设置合理的 perplexity 值
    perplexity_value = min(30, n_samples - 1)  # 确保 perplexity 小于样本数

    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=0)
    transformed_features = tsne.fit_transform(features)

    plt.figure(figsize=(10, 6))
    plt.scatter(transformed_features[:, 0], transformed_features[:, 1], marker='o', alpha=0.7)
    plt.title('t-SNE Visualization of LSTM Features')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()


def plot_lsheatmap(features):
    # 假设features形状为(1, num_timesteps, feature_dim)，去掉批次维度
    features = features.squeeze(0)  # 现在 features 的形状应为(num_timesteps, feature_dim)

    # 确保张量在CPU上并转换为numpy数组
    if torch.is_tensor(features):
        features = features.cpu().detach().numpy()

    plt.figure(figsize=(12, 8))
    sns.heatmap(features, annot=False, cmap='coolwarm')
    plt.title('Heatmap of LSTM Features over Time')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Time Step')
    plt.show()



def animate_lsfeatures(features):
    # 确保张量在CPU上并转换为numpy数组
    if torch.is_tensor(features):
        features = features.cpu().detach().numpy()

    num_timesteps = features.shape[1]
    num_features = features.shape[2]

    # 设置图形和轴
    fig, ax = plt.subplots()
    x = np.arange(num_features)  # 特征的索引
    line, = ax.plot(x, np.random.rand(num_features), 'r-')  # 初始化一条线
    ax.set_ylim(0, 1)  # 假设特征的规模是0到1

    # 更新函数，每帧调用
    def update(frame):
        line.set_ydata(features[0, frame, :])  # 更新线条数据
        return line,

    # 创建动画
    ani = FuncAnimation(fig, update, frames=num_timesteps, blit=True, repeat=False)
    plt.show()





def mytest(video_features):
    # print(video_features)
    # print(video_features[0].shape)
    if isinstance(video_features, list):
        video_features = np.array(video_features)

    print("video_features 维度:", video_features.ndim)
    print("video_features 形状:", video_features.shape)

    # print(video_features.shape)
    result_label.config(text=f"第一个特征向量的大小：{video_features[0].shape}-特征长度：{len(video_features)}")



if __name__ == '__main__':
    # 创建主窗口
    root = tk.Tk()
    root.title("视频动态特征提取及可视化")
    root.geometry("1600x850")  # 设置窗口的初始大小为宽1600像素，高850像素


    # 创建画布
    canvas = tk.Canvas(root)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # 创建滚动条
    scrollbar = ttk.Scrollbar(root, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # 配置画布与滚动条的关联
    canvas.configure(yscrollcommand=scrollbar.set)

    # 设置按钮使用的字体和大小
    custom_font = font.Font(family="Helvetica", size=12)

    CNN_label = Label(root, text="CNN特征提取模块", font=custom_font)
    CNN_label.place(x=0, y=0)

    # #测试按钮 （提取到的特征）
    # button = tk.Button(root, text="输出提取到的特征信息", padx=2, pady=5, width=20, height=1, font=custom_font,
    #                    command=lambda: mytest(video_features))
    # button.place(x=920, y=350)


    # 按钮resnet 18
    button_load_and_extract = tk.Button(root, text="resnet18提取特征", padx=2, pady=5, width=16, height=1,
                                        font=custom_font,
                                        command=use_extract_features_CNN_resnet18)
    button_load_and_extract.place(x=0, y=50)

    button = tk.Button(root, text="CNN特征动画", padx=2, pady=5, width=16, height=1,
                       font=custom_font, command=lambda: feacture_change_dynamics_resnet18(video_features))
    button.place(x=160, y=50)

    button = tk.Button(root, text="PCA降维-主成分分析", padx=2, pady=5, width=18, height=1,
                       font=custom_font, command=lambda: dimension_reduction_pca_resnet18(video_features))
    button.place(x=160 * 2, y=50)

    button_load_and_extract = tk.Button(root, text="resnet18提取第一层特征", padx=2, pady=5, width=20, height=1,
                                        font=custom_font,
                                        command=use_extract_features1_CNN_resnet18)
    button_load_and_extract.place(x=160*3+17, y=50)

    button = tk.Button(root, text="第一层主成分分析", padx=2, pady=5, width=16, height=1,
                       font=custom_font, command=lambda: dimension_reduction_pca1_resnet18(video_features))
    button.place(x=160* 4+53, y=50)

    button_load_and_extract = tk.Button(root, text="resnet18提取第二层特征", padx=2, pady=5, width=20, height=1,
                                        font=custom_font,
                                        command=use_extract_features2_CNN_resnet18)
    button_load_and_extract.place(x=160 * 5+53, y=50)

    button = tk.Button(root, text="第二层主成分分析", padx=2, pady=5, width=15, height=1,
                       font=custom_font, command=lambda: dimension_reduction_pca2_resnet18(video_features))
    button.place(x=160 * 6+88, y=50)

    button_load_and_extract = tk.Button(root, text="resnet18提取第三层特征", padx=2, pady=5, width=20, height=1,
                                        font=custom_font,
                                        command=use_extract_features3_CNN_resnet18)
    button_load_and_extract.place(x=160 * 7+78, y=50)

    button = tk.Button(root, text="第三层主成分分析", padx=2, pady=5, width=15, height=1,
                       font=custom_font, command=lambda: dimension_reduction_pca3_resnet18(video_features))
    button.place(x=160 * 8+113, y=50)



    # 按钮resnet 50
    button_load_and_extract = tk.Button(root, text="resnet50提取特征", padx=2, pady=5, width=16, height=1,
                                        font=custom_font,
                                        command=use_extract_features_CNN_resnet50)
    button_load_and_extract.place(x=0, y=100)

    button = tk.Button(root, text="CNN特征动画", padx=2, pady=5, width=16, height=1,
                       font=custom_font, command=lambda: feacture_change_dynamics_resnet50(video_features))
    button.place(x=160, y=100)

    button = tk.Button(root, text="PCA降维-主成分分析", padx=2, pady=5, width=18, height=1,

                       font=custom_font, command=lambda: dimension_reduction_pca_resnet50(video_features))
    button.place(x=160 * 2, y=100)

    button_load_and_extract = tk.Button(root, text="resnet50提取第一层特征", padx=2, pady=5, width=20, height=1,
                                        font=custom_font,
                                        command=use_extract_features1_CNN_resnet50)
    button_load_and_extract.place(x=160 * 3 + 17, y=100)

    button = tk.Button(root, text="第一层主成分分析", padx=2, pady=5, width=16, height=1,
                       font=custom_font, command=lambda: dimension_reduction_pca1_resnet50(video_features))
    button.place(x=160 * 4 + 53, y=100)

    button_load_and_extract = tk.Button(root, text="resnet50提取第二层特征", padx=2, pady=5, width=20, height=1,
                                        font=custom_font,
                                        command=use_extract_features2_CNN_resnet50)
    button_load_and_extract.place(x=160 * 5 + 53, y=100)

    button = tk.Button(root, text="第二层主成分分析", padx=2, pady=5, width=15, height=1,
                       font=custom_font, command=lambda: dimension_reduction_pca2_resnet50(video_features))
    button.place(x=160 * 6 + 88, y=100)

    button_load_and_extract = tk.Button(root, text="resnet50提取第三层特征", padx=2, pady=5, width=20, height=1,
                                        font=custom_font,
                                        command=use_extract_features3_CNN_resnet50)
    button_load_and_extract.place(x=160 * 7 + 78, y=100)

    button = tk.Button(root, text="第三层主成分分析", padx=2, pady=5, width=15, height=1,
                       font=custom_font, command=lambda: dimension_reduction_pca3_resnet50(video_features))
    button.place(x=160 * 8 + 113, y=100)


    # 按钮resnet 101
    button_load_and_extract = tk.Button(root, text="resnet101提取特征", padx=2, pady=5, width=16, height=1,
                                        font=custom_font,
                                        command=use_extract_features_CNN_resnet101)
    button_load_and_extract.place(x=0, y=150)

    button = tk.Button(root, text="CNN特征动画", padx=2, pady=5, width=16, height=1,
                       font=custom_font, command=lambda: feacture_change_dynamics_resnet101(video_features))
    button.place(x=160, y=150)

    button = tk.Button(root, text="PCA降维-主成分分析", padx=2, pady=5, width=18, height=1,

                       font=custom_font, command=lambda: dimension_reduction_pca_resnet101(video_features))
    button.place(x=160 * 2, y=150)

    button_load_and_extract = tk.Button(root, text="resnet101提取第一层特征", padx=2, pady=5, width=20, height=1,
                                        font=custom_font,
                                        command=use_extract_features1_CNN_resnet101)
    button_load_and_extract.place(x=160 * 3 + 17, y=150)

    button = tk.Button(root, text="第一层主成分分析", padx=2, pady=5, width=16, height=1,
                       font=custom_font, command=lambda: dimension_reduction_pca1_resnet101(video_features))
    button.place(x=160 * 4 + 53, y=150)

    button_load_and_extract = tk.Button(root, text="resnet101提取第二层特征", padx=2, pady=5, width=20, height=1,
                                        font=custom_font,
                                        command=use_extract_features2_CNN_resnet101)
    button_load_and_extract.place(x=160 * 5 + 53, y=150)

    button = tk.Button(root, text="第二层主成分分析", padx=2, pady=5, width=15, height=1,
                       font=custom_font, command=lambda: dimension_reduction_pca2_resnet101(video_features))
    button.place(x=160 * 6 + 88, y=150)

    button_load_and_extract = tk.Button(root, text="resnet101提取第三层特征", padx=2, pady=5, width=20, height=1,
                                        font=custom_font,
                                        command=use_extract_features3_CNN_resnet101)
    button_load_and_extract.place(x=160 * 7 + 78, y=150)

    button = tk.Button(root, text="第三层主成分分析", padx=2, pady=5, width=15, height=1,
                       font=custom_font, command=lambda: dimension_reduction_pca3_resnet101(video_features))
    button.place(x=160 * 8 + 113, y=150)


    # 按钮resnet 152
    button_load_and_extract = tk.Button(root, text="resnet152提取特征", padx=2, pady=5, width=16, height=1,
                                        font=custom_font,
                                        command=use_extract_features_CNN_resnet152)
    button_load_and_extract.place(x=0, y=200)

    button = tk.Button(root, text="CNN特征动画", padx=2, pady=5, width=16, height=1,
                       font=custom_font, command=lambda: feacture_change_dynamics_resnet152(video_features))
    button.place(x=160, y=200)

    button = tk.Button(root, text="PCA降维-主成分分析", padx=2, pady=5, width=18, height=1,

                       font=custom_font, command=lambda: dimension_reduction_pca_resnet152(video_features))
    button.place(x=160 * 2, y=200)

    button_load_and_extract = tk.Button(root, text="resnet152提取第一层特征", padx=2, pady=5, width=20, height=1,
                                        font=custom_font,
                                        command=use_extract_features1_CNN_resnet152)
    button_load_and_extract.place(x=160 * 3 + 17, y=200)

    button = tk.Button(root, text="第一层主成分分析", padx=2, pady=5, width=16, height=1,
                       font=custom_font, command=lambda: dimension_reduction_pca1_resnet152(video_features))
    button.place(x=160 * 4 + 53, y=200)

    button_load_and_extract = tk.Button(root, text="resnet152提取第二层特征", padx=2, pady=5, width=20, height=1,
                                        font=custom_font,
                                        command=use_extract_features2_CNN_resnet152)
    button_load_and_extract.place(x=160 * 5 + 53, y=200)

    button = tk.Button(root, text="第二层主成分分析", padx=2, pady=5, width=15, height=1,
                       font=custom_font, command=lambda: dimension_reduction_pca2_resnet152(video_features))
    button.place(x=160 * 6 + 88, y=200)

    button_load_and_extract = tk.Button(root, text="resnet152提取第三层特征", padx=2, pady=5, width=20, height=1,
                                        font=custom_font,
                                        command=use_extract_features3_CNN_resnet152)
    button_load_and_extract.place(x=160 * 7 + 78, y=200)

    button = tk.Button(root, text="第三层主成分分析", padx=2, pady=5, width=15, height=1,
                       font=custom_font, command=lambda: dimension_reduction_pca3_resnet152(video_features))
    button.place(x=160 * 8 + 113, y=200)


    # 按钮resnet 200
    button_load_and_extract = tk.Button(root, text="resnet200提取特征", padx=2, pady=5, width=16, height=1,
                                        font=custom_font,
                                        command=use_extract_features_CNN_resnet200)
    button_load_and_extract.place(x=0, y=250)

    button = tk.Button(root, text="CNN特征动画", padx=2, pady=5, width=16, height=1,
                       font=custom_font, command=lambda: feacture_change_dynamics_resnet200(video_features))
    button.place(x=160, y=250)

    button = tk.Button(root, text="PCA降维-主成分分析", padx=2, pady=5, width=18, height=1,

                       font=custom_font, command=lambda: dimension_reduction_pca_resnet200(video_features))
    button.place(x=160 * 2, y=250)

    button_load_and_extract = tk.Button(root, text="resnet200提取第一层特征", padx=2, pady=5, width=20, height=1,
                                        font=custom_font,
                                        command=use_extract_features1_CNN_resnet200)
    button_load_and_extract.place(x=160 * 3 + 17, y=250)

    button = tk.Button(root, text="第一层主成分分析", padx=2, pady=5, width=16, height=1,
                       font=custom_font, command=lambda: dimension_reduction_pca1_resnet200(video_features))
    button.place(x=160 * 4 + 53, y=250)

    button_load_and_extract = tk.Button(root, text="resnet200提取第二层特征", padx=2, pady=5, width=20, height=1,
                                        font=custom_font,
                                        command=use_extract_features2_CNN_resnet200)
    button_load_and_extract.place(x=160 * 5 + 53, y=250)

    button = tk.Button(root, text="第二层主成分分析", padx=2, pady=5, width=15, height=1,
                       font=custom_font, command=lambda: dimension_reduction_pca2_resnet200(video_features))
    button.place(x=160 * 6 + 88, y=250)

    button_load_and_extract = tk.Button(root, text="resnet200提取第三层特征", padx=2, pady=5, width=20, height=1,
                                        font=custom_font,
                                        command=use_extract_features3_CNN_resnet200)
    button_load_and_extract.place(x=160 * 7 + 78, y=250)

    button = tk.Button(root, text="第三层主成分分析", padx=2, pady=5, width=15, height=1,
                       font=custom_font, command=lambda: dimension_reduction_pca3_resnet200(video_features))
    button.place(x=160 * 8 + 113, y=250)


    # 按钮VGG16
    button_load_and_extract = tk.Button(root, text="VGG16提取特征", padx=2, pady=5, width=18, height=1, font=custom_font,
                                        command=use_extract_features_VGG16)
    button_load_and_extract.place(x=0, y=300)

    button = tk.Button(root, text="CNN特征动画", padx=2, pady=5, width=18, height=1, font=custom_font,
                       command=lambda: feature_change_dynamics_VGG16(video_features))
    button.place(x=220, y=300)

    button = tk.Button(root, text="PCA降维-主成分分析", padx=2, pady=5, width=18, height=1, font=custom_font,
                       command=lambda: dimension_reduction_pca_first_layer(video_features))
    button.place(x=440, y=300)

    # button_load_and_extract = tk.Button(root, text="提取第一层特征", padx=2, pady=5, width=20, height=1,
    #                                     font=custom_font,
    #                                     command=extract_first_layer_features_VGG16)
    # button_load_and_extract.place(x=480+17, y=300)
    #
    # button = tk.Button(root, text="第一层主成分分析", padx=2, pady=5, width=16, height=1, font=custom_font,
    #                    command=lambda: dimension_reduction_pca_first_layer(video_features))
    # button.place(x=640+53, y=300)
    #
    # button_load_and_extract = tk.Button(root, text="提取第二层特征", padx=2, pady=5, width=20, height=1,
    #                                     font=custom_font,
    #                                     command=use_extract_middle_layer_features_VGG16)
    # button_load_and_extract.place(x=800+53, y=300)
    #
    # button = tk.Button(root, text="第二层主成分分析", padx=2, pady=5, width=15, height=1, font=custom_font,
    #                    command=lambda: dimension_reduction_pca_middle_layer(video_features))
    # button.place(x=960+88, y=300)
    #
    # button_load_and_extract = tk.Button(root, text="提取第三层特征", padx=2, pady=5, width=20, height=1,
    #                                     font=custom_font,
    #                                     command=use_extract_last_layer_features_VGG16)
    # button_load_and_extract.place(x=1120+78, y=300)
    #
    # button = tk.Button(root, text="第三层主成分分析", padx=2, pady=5, width=15, height=1, font=custom_font,
    #                    command=lambda: dimension_reduction_pca_last_layer(video_features))
    # button.place(x=1280+113, y=300)

    # 按钮inception

    button_load_and_extract = tk.Button(root, text="inception提取特征", padx=2, pady=5, width=18, height=1, font=custom_font,
                                        command=use_extract_features_inception)
    button_load_and_extract.place(x=0, y=350)

    button = tk.Button(root, text="特征热图", padx=2, pady=5, width=18, height=1,
                       font=custom_font, command=lambda: visualize_features_inception(video_features))
    button.place(x=220, y=350)

    button = tk.Button(root, text="PCA降维-主成分分析", padx=2, pady=5, width=18, height=1, font=custom_font,
                       command=lambda: dimension_reduction_pca_inception(video_features))
    button.place(x=440, y=350)

    # button_load_and_extract = tk.Button(root, text="提取第一层特征", padx=2, pady=5, width=15, height=1,
    #                                     font=custom_font,
    #                                     command=extract_first_layer_features_VGG16)
    # button_load_and_extract.place(x=480, y=350)
    #
    # button = tk.Button(root, text="第一层主成分分析", padx=2, pady=5, width=15, height=1, font=custom_font,
    #                    command=lambda: dimension_reduction_pca_first_layer(video_features))
    # button.place(x=640, y=350)
    #
    # button_load_and_extract = tk.Button(root, text="提取第二层特征", padx=2, pady=5, width=15, height=1,
    #                                     font=custom_font,
    #                                     command=use_extract_middle_layer_features_VGG16)
    # button_load_and_extract.place(x=800, y=350)
    #
    # button = tk.Button(root, text="第二层主成分分析", padx=2, pady=5, width=15, height=1, font=custom_font,
    #                    command=lambda: dimension_reduction_pca_middle_layer(video_features))
    # button.place(x=960, y=350)
    #
    # button_load_and_extract = tk.Button(root, text="提取第三层特征", padx=2, pady=5, width=15, height=1,
    #                                     font=custom_font,
    #                                     command=use_extract_last_layer_features_VGG16)
    # button_load_and_extract.place(x=1120, y=350)
    #
    # button = tk.Button(root, text="第三层主成分分析", padx=2, pady=5, width=15, height=1, font=custom_font,
    #                    command=lambda: dimension_reduction_pca_last_layer(video_features))
    # button.place(x=1280, y=350)


    # 按钮densenet

    button_load_and_extract = tk.Button(root, text="densenet提取特征", padx=2, pady=5, width=18, height=1,
                                        font=custom_font,
                                        command=use_extract_features_CNN_densenet121)
    button_load_and_extract.place(x=0, y=400)

    button = tk.Button(root, text="特征热图", padx=2, pady=5, width=18, height=1,
                       font=custom_font, command=lambda: visualize_features_densenet(video_features))
    button.place(x=220, y=400)

    button = tk.Button(root, text="densenet特征动画", padx=2, pady=5, width=18, height=1,
                       font=custom_font, command=lambda: feature_change_dynamics_densenet121(video_features))
    button.place(x=440, y=400)

    # button_load_and_extract = tk.Button(root, text="提取第一层特征", padx=2, pady=5, width=15, height=1,
    #                                     font=custom_font,
    #                                     command=extract_first_layer_features_VGG16)
    # button_load_and_extract.place(x=480, y=350)
    #
    # button = tk.Button(root, text="第一层主成分分析", padx=2, pady=5, width=15, height=1, font=custom_font,
    #                    command=lambda: dimension_reduction_pca_first_layer(video_features))
    # button.place(x=640, y=400)
    #
    # button_load_and_extract = tk.Button(root, text="提取第二层特征", padx=2, pady=5, width=15, height=1,
    #                                     font=custom_font,
    #                                     command=use_extract_middle_layer_features_VGG16)
    # button_load_and_extract.place(x=800, y=400)
    #
    # button = tk.Button(root, text="第二层主成分分析", padx=2, pady=5, width=15, height=1, font=custom_font,
    #                    command=lambda: dimension_reduction_pca_middle_layer(video_features))
    # button.place(x=960, y=400)
    #
    # button_load_and_extract = tk.Button(root, text="提取第三层特征", padx=2, pady=5, width=15, height=1,
    #                                     font=custom_font,
    #                                     command=use_extract_last_layer_features_VGG16)
    # button_load_and_extract.place(x=1120, y=400)
    #
    # button = tk.Button(root, text="第三层"
    #                               "主成分分析", padx=2, pady=5, width=15, height=1, font=custom_font,
    #                    command=lambda: dimension_reduction_pca_last_layer(video_features))
    # button.place(x=1280, y=400)


#C3D
    CNN_label = Label(root, text="C3D特征提取模块", font=custom_font)
    CNN_label.place(x=0, y=500)

    button_load_and_extract = tk.Button(root, text="C3D提取特征", padx=2, pady=5, width=18, height=1,
                                        font=custom_font,
                                        command=load_video_and_extract_3dfeatures)
    button_load_and_extract.place(x=0, y=550)

    button = tk.Button(root, text="3D特征图可视化", padx=2, pady=5, width=18, height=1,
                       font=custom_font, command=lambda: visualize_feature_maps(video_features))
    button.place(x=220, y=550)

    button = tk.Button(root, text="热图", padx=2, pady=5, width=18, height=1,
                       font=custom_font, command=lambda: visualize_heatmaps(video_features))
    button.place(x=440, y=550)

    button = tk.Button(root, text="直方图", padx=2, pady=5, width=18, height=1,
                       font=custom_font, command=lambda: visualize_feature_histograms(video_features))
    button.place(x=0, y=600)

    # #你可以通过绘制特征向量中值的分布直方图来分析模型学习到的特征的统计属性。这对于理解模型在各个层次上的行为非常有帮助，特别是了解数据在模型中如何变化。
    button = tk.Button(root, text="特征图动态显示", padx=2, pady=5, width=18, height=1,
                       font=custom_font, command=lambda: animate_feature_maps(video_features))
    button.place(x=220, y=600)

    # #如果你的特征包括时间维度，可以通过动态图形（例如动画）展示随时间变化的特征图，以更直观地理解模型如何处理视频序列。
    button = tk.Button(root, text="热图动态显示", padx=2, pady=5, width=18, height=1,
                       font=custom_font, command=lambda: animate_heatmaps(video_features))
    button.place(x=440, y=600)




#LSTM
    CNN_label = Label(root, text="LSTM特征提取模块", font=custom_font)
    CNN_label.place(x=0, y=700)

    # #类似于特征图动态展示，你也可以将热图随时间变化展示出来，这有助于分析模型对视频中不同时间点的响应。
    button_load_and_extract = tk.Button(root, text="提取第一层lstm特征", padx=2, pady=5, width=18, height=1, font=custom_font,
                                        command=load_video_and_extract_multilayer_lstm1_features)
    button_load_and_extract.place(x=0, y=750)

    button_load_and_extract = tk.Button(root, text="提取第二层lstm特征", padx=2, pady=5, width=18, height=1, font=custom_font,
                                        command=load_video_and_extract_multilayer_lstm2_features)
    button_load_and_extract.place(x=220, y=750)

    button_load_and_extract = tk.Button(root, text="提取第三层lstm特征", padx=2, pady=5, width=18, height=1, font=custom_font,
                                        command=load_video_and_extract_multilayer_lstm3_features)
    button_load_and_extract.place(x=440, y=750)

    #
    button = tk.Button(root, text="t-SNE降维", padx=2, pady=5, width=18, height=1,
                       font=custom_font, command=lambda: visualize_lsfeaturesls(video_features))
    button.place(x=0, y=800)
    # # 由于LSTM输出的特征维度可能很高，可以使用降维技术如PCA（主成分分析）或t-SNE来降低特征的维度，然后将这些或三维空间中。这可以帮助我们观察低维特征绘制在二维不同帧或视频片段的特征如何聚类，从而理解哪些帧是相似的

    button = tk.Button(root, text="时间—特征维度", padx=2, pady=5, width=18, height=1,
                       font=custom_font, command=lambda: plot_lsheatmap(video_features))
    button.place(x=220, y=800)

    # #你感兴趣的是单个视频序列的时间动态，可以将LSTM的每一步输出（或隐藏状态）绘制为热图，其中行表示时间步，列表示特征维度
    #
    button = tk.Button(root, text="每帧特征变化", padx=2, pady=5, width=18, height=1,
                       font=custom_font, command=lambda: animate_lsfeatures(video_features))
    button.place(x=440, y=800)
    # #LSTM的输出以动画形式展示每一帧对应的特征变化也是一个直观的方法。可以使用matplotlib.animation


    # 添加标签用于显示结果
    result_label = Label(root, text="请加载视频文件以提取特征", font=custom_font)
    result_label.place(x=800, y=500)

    # 添加进度条
    progress_bar = ttk.Progressbar(root, orient="horizontal", length=500, mode="determinate")
    progress_bar.place(x=800, y=550)

    # 测试按钮 （提取到的特征）
    button = tk.Button(root, text="输出提取到的特征信息", padx=2, pady=5, width=20, height=1, font=custom_font,
                       command=lambda: mytest(video_features))
    button.place(x=920, y=650)

    # 进入主事件循环
    root.mainloop()


