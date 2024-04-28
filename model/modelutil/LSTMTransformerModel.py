import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from component.MyLstm import LSTMModel
from component.MyTransformer import TransformerModel


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 
input_dim = 22

class MyData(Dataset):
    def __init__(self, data_path,ruler_path):
        self.file_names = []
        self.labels = []
        self.datas = []
        label_list = ['congest', 'malicious', 'out', 'nodedown', 'normal', 'obstacle', 'appdown']
        # 遍历所有文件夹并获取文件名及其标签
        for label in label_list:
            folder_path = os.path.join(data_path, label)
            r_path = os.path.join(ruler_path, label)
            for file in os.listdir(folder_path):
                if file.endswith(".csv"):
                    file_path1 = os.path.join(folder_path, file)
                    file_path2 = os.path.join(r_path, file)
                    if label == 'normal':
                        binary_label = 0
                    else:
                        binary_label = 1
                    self.datas.append((file_path1,file_path2, label_list.index(label), binary_label))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        dir1,dir2, label, binary_label = self.datas[index]
        math = pd.read_csv(dir1, header=None).values.astype(np.float32)
        ruler = pd.read_csv(dir2, header=None).values.astype(np.float32)
        data = torch.tensor(math)
        label = torch.tensor(label, dtype=torch.int64)
        binary_label = torch.tensor(binary_label, dtype=torch.float32)
        bn = torch.nn.BatchNorm1d(input_dim, affine=False)
        data = bn(data)
        return data,ruler, label, binary_label
    
class LstmTransformer(nn.Module):
    def __init__(self,learning_rate,output_size,numclasses=7,hidden_size=128):
        # 50 200 0.0002 2 69.31 85.03
        self.learning_rate = learning_rate
        super(LstmTransformer, self).__init__()
        self.transformer = TransformerModel(output_size)
        self.lstm = LSTMModel()
        self.line1 = nn.Linear(hidden_size+27,hidden_size)
        self.line2 = nn.Linear(hidden_size,int(hidden_size/4))
        self.class1 = nn.Linear(int(hidden_size/4),numclasses)
        self.dect = nn.Linear(int(hidden_size/4),2)

    def forward(self,x ,rulers): 
        # x.size():[40, 96, 22]
        # x = x.permute(0, 2, 1)  
        rulers = rulers.view(-1,27)
        # lstm_output.size():40,128
        # x.size():[40, 22, 96]
        transformer_output = self.transformer(x)
        # transformer_output.size():[40, 22, 96]
        lstm_output = self.lstm(transformer_output)
        # lstm_output.size():[40,128]
        combined_output = torch.cat((lstm_output, rulers), dim=1)
        x = self.line1(combined_output)
        x = self.line2(x)
        classT = self.class1(x)
        dectT = self.dect(x)
        return classT,dectT

def run(math_path,ruler_path,learning_rate,batch_size,epoch_num,model_path,init=True):
    trainData = MyData(math_path,ruler_path)
    trainLoader = torch.utils.data.DataLoader(trainData, batch_size, shuffle=True)
    model = LstmTransformer(learning_rate,22) 
    if False==init:
        model_state_dict = torch.load(model_path , map_location=device)  
        model.load_state_dict(model_state_dict)  
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    classAcurList = []
    dectAcurList = []
    for epoch in range(epoch_num): 
        epoch_class_corrects = 0  
        epoch_dect_corrects = 0
        epoch_samples = 0  
        for i, (datas,rulers, labels, blabels) in enumerate(trainLoader):            
            classOutput,dectOutput = model(datas,rulers)
            blabels = blabels.long()
            classLoss = criterion(classOutput,labels )
            dectLoss = criterion(dectOutput,blabels )
            optimizer.zero_grad()
            loss = classLoss + dectLoss
            loss.backward()
            optimizer.step()
            # 计算精度  
            _, classPreds = torch.max(classOutput, 1)  # 获取预测类别  
            class_corrects = (classPreds == blabels).sum().item()  
            epoch_class_corrects += class_corrects  
            _, dectPreds = torch.max(dectOutput, 1)  # 获取预测类别  
            dect_corrects = (dectPreds == labels).sum().item()  
            epoch_dect_corrects += dect_corrects  
            epoch_samples += labels.size(0)
        class_accuracy = 100.0 * epoch_class_corrects / epoch_samples  
        classAcurList.append(class_accuracy)  
        # 如果你也需要计算dectOutput的精度  
        dect_accuracy = 100.0 * epoch_dect_corrects / epoch_samples  
        dectAcurList.append(dect_accuracy)  
        torch.save(model.state_dict(), f'{model_path}/model{epoch+1}.pkl')
        print(f'{epoch+1},{classLoss.item()},{dectLoss.item()},{class_accuracy},{dect_accuracy}')
