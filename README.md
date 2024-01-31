# 多模态情感分析

## 实验介绍
本实验旨在通过结合文本和图片信息，进行多模态情感分析的实验。

## 实验环境
本次实验基于python 3.10.2，要运行代码，需要以下依赖项：
* matplotlib==3.7.1
* pandas==2.0.3
* Pillow==9.5.0
* scikit-learn==1.3.0
* torch==2.1.1+cu118
* torchvision==0.16.1+cu118
* tqdm==4.66.1
* transformers==4.34.0

可以通过运行下面的代码进行环境配置 
```python
pip install -r requirements.txt
```


## 文件结构
```python
├── dataset
    ├── data/ #由于数据文件过大，并未上传至GitHub上
    ├── test_without_label.txt #测试数据
    ├── test_with_pred_label.txt #最终预测结果
    └── train.txt #训练数据的guid和对应的情感标签
├── dataloader
    ├── test_dataloader.pth
    ├── train_dataloader.pth
    └── val_dataloader.pth
├── model_best.pth #保存的最佳多模态模型
├── model_best_option_0.pth #保存的只有图像数据输入时的最佳模型
├── model_best_option_1.pth #保存的只有文本数据输入时的最佳模型
├── model_best_option_2.pth #保存的图像和文本输入的双模态融合最佳模型，其实跟model_best.pth相同
├── README.md
├── requirements.txt
├── model.py #模型代码
├── data_proce.py #数据处理
├── test.py 
├── train.py 
├── bert-base-multilingual-cased/ #预训练BERT模型
```
## 数据文件
data文件中包含着txt和image文件，虽未上传至此平台，如果需要，仍可以通过链接获取。
```
链接：https://pan.baidu.com/s/1geKBqTw96gOFqSm3V_OpHg?pwd=p9vy 
提取码：p9vy 
--来自百度网盘超级会员V5的分享
```

## 代码执行流程

### 1. 数据预处理
将数据进行预处理，并且存储为 DataLoade
```python
python data_process.py --batch_size BATCH_SIZE --num_workers NUM_WORKERS
```

其中:
```
BATCH_SIZE: DataLoader 的批处理大小，默认为 64
NUM_WORKERS: DataLoader 的工作线程数，默认为 4
```
执行此命令后将在`dataset/dataloader`目录下生成三个文件:
```
train_dataloader.pth
val_dataloader.pth
test_dataloader.pth
```


### 2. 模型训练与存储
通过训练集进行模型训练，验证集对模型调参验证，并对最佳模型存储
```python
python train.py --epochs EPOCHS --lr LR
```

其中:
```
EPOCHS: 训练的轮数，默认为 10
LR: 学习率，默认为 1e-4
```

由于代码中默认是多模态，执行此命令将使用训练和验证模型，并在每个轮次中对比生成并保存最佳多模态模型:
```
mode_best.pth
```
同时，执行此命令将使用训练和验证模型，并在每个轮次中对比生成并保存只有图像数据输入、只有文本数据输入、图像和文本输入的双模态融合的三个最佳模型:
```
model_best_option_0.pth #保存的只有图像数据输入时的最佳模型
model_best_option_1.pth #保存的只有文本数据输入时的最佳模型
model_best_option_2.pth #保存的图像和文本输入的双模态融合最佳模型，其实跟model_best.pth相同
```

### 3. 测试集预测
利用最佳模型对测试集进行预测相应的的情感标签
```python
python test.py
```

此命令将使用存储的最佳模型`model_best.pth`对测试集进行预测，并在`dataset`文件夹中生成一个名为`test_with_pred_label.txt`的文件。

**注意**：
1. 用于文本预训练的bert模型是从Hugging Face Transformers库(https://huggingface.co/bert-base-multilingual-cased)下载的。`bert-base-multilingual-cased`文件下是bert模型的配置。
2. 请尽量不要打乱文件的结构，这可能会影响代码的运行。
