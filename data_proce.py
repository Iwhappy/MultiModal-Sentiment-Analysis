import os
import pandas as pd
import torch
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer


class MultiModalDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        guid = self.df.iloc[idx, 0]
        
        img_path = os.path.join(self.img_dir, str(guid) + '.jpg')
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        text_path = os.path.join(self.img_dir, str(guid) + '.txt')
        with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().strip()
        label = self.df.iloc[idx, 1]

        return img, text, label

def collate_fn(batch):
    imgs, texts, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    # use BertTokenizer to tokenize the text
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    texts, atten_masks, token_type_ids = tokenizer(list(texts), padding=True, truncation=True, return_tensors='pt').values()
    atten_masks = atten_masks
    token_type_ids = token_type_ids
    tmp_labels = []
    for label in labels:
        if label == 'positive':
            tmp_labels.append(0)
        elif label == 'neutral':
            tmp_labels.append(1)
        else:
            tmp_labels.append(2)

    labels = torch.tensor(tmp_labels)
    return imgs, texts, atten_masks, token_type_ids, labels

def get_args():
    try:
        # 尝试解析命令行参数
        args = argparse.ArgumentParser()
        # dataloader 的 batch size
        args.add_argument('--batch_size', type=int, default=16)
        args.add_argument('--num_workers', type=int, default=4)  # dataloader 的工作进程数
        return args.parse_args()
    except:
        # 处理 Jupyter 特定的参数
        return argparse.Namespace(batch_size=16, num_workers=4)


if __name__ == '__main__':

    args = get_args()

    # load train.txt
    train_df = pd.read_csv('./dataset/train.txt', sep=',')
    test_df = pd.read_csv('./dataset/test_without_label.txt', sep=',')

    # split train_df into train_df and val_df
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['tag'])

    print('train_df shape: ', train_df.shape)
    print('val_df shape: ', val_df.shape)
    print('test_df shape: ', test_df.shape)

    # image transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # train dataloader
    train_dataset = MultiModalDataset(train_df, './dataset/data', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # val dataloader
    val_dataset = MultiModalDataset(val_df, './dataset/data', transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # test dataloader
    test_dataset = MultiModalDataset(test_df, './dataset/data', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # 保存 dataloader
    save_dir = './dataloader'
    os.makedirs(save_dir, exist_ok=True)  # 创建目录（如果不存在）

    torch.save(train_dataloader, os.path.join(save_dir, 'train_dataloader.pth'))
    torch.save(val_dataloader, os.path.join(save_dir, 'val_dataloader.pth'))
    torch.save(test_dataloader, os.path.join(save_dir, 'test_dataloader.pth'))
