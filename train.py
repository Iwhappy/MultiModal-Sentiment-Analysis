import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
from model import MultiModalModel
from data_proce import MultiModalDataset, collate_fn
import pandas as pd

def train_model(epoch, model, train_dataloader, val_dataloader, criterion, optimizer, device):
    # train
    model.train()
    print('start training for epoch {}'.format(epoch))
    
    total_correct = 0 
    running_loss = 0
    accumulation_steps = 3
    # 在训练循环中添加梯度累积
    for batch_index, (imgs, texts, atten_masks, token_type_ids, labels) in tqdm(enumerate(train_dataloader)):
        imgs = imgs.to(device)
        texts = texts.to(device)
        atten_masks = atten_masks.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(imgs, texts, atten_masks, token_type_ids)
        _, preds = torch.max(output, 1)
        loss = criterion(output, labels)
        total_correct += torch.sum(preds == labels).item()
        loss.backward()
        running_loss += loss.item()

        if (batch_index + 1) % accumulation_steps == 0:  # 更新梯度
            optimizer.step()
            optimizer.zero_grad()

        if (batch_index + 1) % 10 == 0:
            print('Epoch {}, Batch {}, Loss: {:.4f}'.format(epoch, batch_index + 1, loss.item()))

        torch.cuda.empty_cache()

    train_loss = running_loss / len(train_dataloader)
    train_acc = total_correct.item() / len(train_dataloader.dataset)
    print('Epoch {}, Training Loss: {:.4f}, Training Accuracy: {:.4f}'.format(epoch, train_loss, train_acc))

    # val
    model.eval()
    print('开始验证第 {} 个 epoch'.format(epoch))
    val_correct = 0
    running_val_loss = 0.0
    with torch.no_grad():
        for batch_index, (imgs, texts, atten_masks, token_type_ids, labels) in tqdm(enumerate(val_dataloader)):
            imgs = imgs.to(device)
            texts = texts.to(device)
            atten_masks = atten_masks.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            output = model(imgs, texts, atten_masks, token_type_ids)
            loss = criterion(output, labels)

            pred = torch.argmax(output, dim=1)
            val_correct += torch.sum(pred == labels).item()
            running_val_loss += loss.item()
        val_acc = val_correct / len(val_dataloader.dataset)
        val_loss = running_val_loss / len(train_dataloader)
        print('Epoch {}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(epoch, val_loss, val_acc))

    return train_loss, train_acc, val_loss, val_acc 

def predict_model(model, test_dataloader, device):
    model.eval()
    pred_list = []

    with torch.no_grad():
        for batch_index, (imgs, texts, atten_masks, token_type_ids, labels) in tqdm(enumerate(test_dataloader)):
            imgs = imgs.to(device)
            texts = texts.to(device)
            atten_masks = atten_masks.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device) 

            output = model(imgs, texts, atten_masks, token_type_ids)
            pred = torch.argmax(output, dim=1)
            pred_list += pred.tolist()

    return pred_list

def get_args2():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', type=int, default=10)  # epochs for training
        parser.add_argument('--lr', type=float, default=1e-4)  # learning rate
        return parser.parse_args()
    except SystemExit:
        # 在 Jupyter 中提供默认参数
        return argparse.Namespace(epochs=10, lr=1e-4)
    
if __name__ == '__main__':

    config = get_args2()

    save_model_dir = ''
    os.makedirs(save_model_dir, exist_ok=True)

    # load dataloader (saved in ./dataloader)
    train_dataloader = torch.load('./dataloader/train_dataloader.pth')
    val_dataloader = torch.load('./dataloader/val_dataloader.pth')
    
    num_classes = 3
    option = 2 
    
    # load model
    model = MultiModalModel(num_classes,option)
    # train
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))
    model.to(device)

    train_acc_list = []
    val_acc_list = []

    # 确保使用 config.epochs 和 config.lr
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc ,val_loss, val_acc= train_model(epoch, model, train_dataloader, val_dataloader, criterion, optimizer, device)
        
        if len(val_acc_list) == 0 or val_acc > max(val_acc_list):
            torch.save(model.state_dict(), os.path.join(save_model_dir, 'model_best.pth'))
        val_acc_list.append(val_acc)
        train_acc_list.append(train_acc)

    print('best acc: {}'.format(max(val_acc_list)))

    plt.plot(range(1, config.epochs + 1), train_acc_list, label='Train Acc')
    plt.plot(range(1, config.epochs + 1), val_acc_list, label='Validation Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./多模态.png')
    plt.show()


    option_values = [0, 1, 2]#分别对应：仅图像、仅文本、双模态融合
    plt.figure(figsize=(10, 6))

    for option in option_values:
        # 重新加载模型和数据加载器
        model = MultiModalModel(num_classes, option)

        val_acc_list = []
        train_acc_list = []

        for epoch in range(1, config.epochs + 1):
            train_loss, train_acc, val_loss, val_acc = train_model(epoch, model, train_dataloader, val_dataloader, criterion, optimizer, device)
            
            if len(val_acc_list) == 0 or val_acc > max(val_acc_list):
                torch.save(model.state_dict(), os.path.join(save_model_dir, f'model_best_option_{option}.pth'))
            val_acc_list.append(val_acc)
            train_acc_list.append(train_acc)

        # 绘制曲线
        plt.plot(range(1, config.epochs + 1), val_acc_list, label=f'Validation Acc (Option {option})', linestyle='-')
        plt.plot(range(1, config.epochs + 1), train_acc_list, label=f'Train Acc (Option {option})', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./acc_消融和多模态.png')
    plt.show()
