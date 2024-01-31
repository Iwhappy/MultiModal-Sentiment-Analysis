import torch
import pandas as pd
import matplotlib.pyplot as plt
from model import MultiModalModel
from train import predict_model


if __name__ == '__main__':
    test_dataloader = torch.load('./dataloader/test_dataloader.pth')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))
    model = MultiModalModel(n_classes=3,option=2)
    model.load_state_dict(torch.load('model_best.pth'))
    model.to(device)
    model.eval()
    print('start test')
    correct = 0
    pred_list = []
    pred_list = predict_model(model, test_dataloader, device)

    # load test data
    test_df = pd.read_csv('./dataset/test_without_label.txt', sep=',')
    # save pred_list
    for i in range(len(test_df)):
        test_df['tag'] = test_df['tag'].astype(str)
        if int(pred_list[i]) == 0:
            test_df.loc[i, 'tag'] = 'positive'
        elif int(pred_list[i]) == 1:
            test_df.loc[i, 'tag'] = 'neutral'
        else:
            test_df.loc[i, 'tag'] = 'negative'

    test_df.to_csv('./dataset/test_with_pred_label.txt', index=False)        
    
    
        