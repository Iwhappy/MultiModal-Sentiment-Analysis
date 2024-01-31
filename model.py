import torch
import torchvision.models as models
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig

# Image 
class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

    def forward(self, input):
        # Input shape: (batch_size, 3, 256, 256)
        output = self.resnet(input)
        return output

# Text 
Bertmodel_folder_path = 'bert-base-multilingual-cased'
# 加载配置文件
Bertconfig = BertConfig.from_pretrained(Bertmodel_folder_path)
# 加载模型权重
Bertmodel = BertModel.from_pretrained(Bertmodel_folder_path, config=Bertconfig)

class TextModel(nn.Module):
    def __init__(self):
        super(TextModel, self).__init__()
        self.bert = Bertmodel

    def forward(self, text, atten_masks, input_ids):
        output = self.bert(text, atten_masks, input_ids)
        pooled_output = output[1]  
        output = pooled_output
        return output

class MultiModalModel(nn.Module):
    def __init__(self, num_classes, option):
        super(MultiModalModel, self).__init__()
        self.option = option
        self.num_classes = num_classes
        
        self.text_model = TextModel()
        self.image_model = ImageModel()

        self.classifier0 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
            nn.ReLU(inplace=True),
           
        )
        self.classifier1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(768, 256),
            #nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
            #nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.classifier2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1768, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
            nn.ReLU(inplace=True),
           # nn.Softmax(dim=1)
        )

    def forward(self, imgs, texts, atten_masks, token_type_ids):

        if(self.option==0):
            image_features = self.image_model(imgs)
            output = image_features
            output = self.classifier0(image_features)

        elif(self.option==1):
            text_features = self.text_model(texts, atten_masks, token_type_ids)
            output = self.classifier1(text_features)

        else:
            image_features = self.image_model(imgs)
            text_features = self.text_model(texts, atten_masks, token_type_ids)
            
            fusion_features = torch.cat((text_features,image_features), dim=-1)
            # print(fusion_features.shape)
            output = self.classifier2(fusion_features)
            # print(output.shape)

        return output