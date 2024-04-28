# -*-coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import csv
import pandas as pd
import numpy as np
import argparse

nNode = 31
# Hyper Parameters
num_e2e = 3
input_size = 18  # 输入数据的维度
length = 96  # 输入数据的长度
num_classes = 7
batch_size = 48
num_epochs = 100

# 训练集
class TrainData(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.datas = []
        self.file_label = {}
        ilabel = 0
        for _, dirs, _ in os.walk(self.data_path):
            for dir in dirs:
                if dir == 'normal':
                    label = 0
                    blabel = 0
                else:
                    ilabel += 1
                    label = ilabel
                    blabel = 1
                for file in os.listdir(self.data_path + dir):
                    if file[-3:] == 'csv':
                        self.datas.append((file, label, blabel))
                self.file_label[label] = dir              
                
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):  
        file, label, blabel = self.datas[index]  
        sample_list = []  
        
        with open(os.path.join(self.data_path, file.split('_')[0], file), 'r') as f:  
            csv_reader = csv.reader(f)  
            for line in csv_reader:  
                row = list(map(float, line))  
                if len(row) < 21:  
                    # 使用某个默认值填充数据  
                    row.extend([0] * (21 - len(row)))  
                elif len(row) > 21:  
                    # 截断多余的数据  
                    row = row[:21]  
                sample_list.append(row)  
    
        # 确保所有样本的长度都是 21  
        assert all(len(s) == 21 for s in sample_list), "All samples should have length 21"  
        
        sample = torch.tensor(sample_list, dtype=torch.float32)  
        label = torch.tensor(label, dtype=torch.long)  # 通常使用 long 类型表示标签  
        blabel = torch.tensor(blabel, dtype=torch.long)  # 同上  
        return sample, label, blabel, file

# 测试集
class TestData(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.datas = {}
        self.file_label = {}
        label = 0
        ilabel = 0
        for _, dirs, _ in os.walk(self.data_path):
            for dir in dirs:
                if dir == 'normal':
                    label = 0
                    blabel = 0
                else:
                    ilabel += 1
                    label = ilabel
                    blabel = 1
                for file in os.listdir(self.data_path + dir):
                    if file[-3:] == 'csv':
                        seed = file.split("_")[1]
                        if seed in self.datas:
                            self.datas[seed].append((file, label, blabel))
                        else:
                            self.datas[seed] = [(file, label, blabel)]
                self.file_label[label] = dir
                label += 1
        self.datas = list(self.datas.values())
                
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        item = []
        for file, label, blabel in self.datas[index]:
            f = open(self.data_path + '/' + file.split('_')[0] + '/' + file, 'r+')
            csv_reader = csv.reader(f)
            sample = []
            for line in csv_reader:
                row = list(map(float, line))
                if len(row) < 21:  
                    # 使用某个默认值填充数据  
                    row.extend([0] * (21 - len(row)))  
                elif len(row) > 21:  
                    # 截断多余的数据  
                    row = row[:21]  
                sample.append(row)
            f.close()

            # sample = torch.tensor(sample, dtype=torch.float32).cuda()
            # label = torch.tensor(label, dtype=torch.int64).cuda()
            # blabel = torch.tensor(blabel, dtype=torch.int64).cuda()
            sample = torch.tensor(sample, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.int64)
            blabel = torch.tensor(blabel, dtype=torch.int64)
            item.append((sample, label, blabel, file))
        return item

# 端到端注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, 1, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(1, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # x 的输入格式是：[batch_size, C, H, W]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # keepdim=True:保持维度不变
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # max_out：最大值，_：最大值的索引
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# 
class LSTM(nn.Module):
    def __init__(self):
        # 50 200 0.0002 2 69.31 85.03
        self.learning_rate = learning_rate
        super(LSTM, self).__init__()
        self.num_layers = 2
        # self.conv_size = 54
        # self.conv_size = 48   
        self.conv_size = 21        
        self.hidden_size = 64
        self.channel_attention = ChannelAttention(3)
        self.spacial_attention = SpatialAttention()
        # self.linear1 = nn.Linear(input_size, 16)
        self.conv = nn.Sequential(
            nn.Conv2d(num_e2e, 1, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(1),
            )
        self.lstm = nn.LSTM(self.conv_size, self.hidden_size, self.num_layers, batch_first=True)  # batch_first=True仅仅针对输入而言
        self.linear1 = nn.Linear(self.hidden_size, int(self.hidden_size / 2))
        self.linear2 = nn.Linear(int(self.hidden_size / 2), int(self.hidden_size / 4))
        self.linearDiog = nn.Linear(int(self.hidden_size / 4), num_classes)
        self.linearDect = nn.Linear(int(self.hidden_size / 4), 2)
        self.normDiog = nn.BatchNorm1d(1)
        self.normDect = nn.BatchNorm1d(1)

    def forward(self, x): 
        new = x
        h0 = Variable(torch.randn(self.num_layers, new.size(0), self.hidden_size))
        c0 = Variable(torch.randn(self.num_layers, new.size(0), self.hidden_size))
        # # Forward propagate RNN
        out, _ = self.lstm(new, (h0, c0))  # 送入一个初始的x值，作为输入以及(h0, c0)
        out = out[:, -1, :]
        out = self.linear1(out)
        out = self.linear2(out)
        out_type = self.linearDiog(out)
        out_type = out_type.view(out_type.shape[0], 1, num_classes)
        out_type = self.normDiog(out_type)
        out_type = out_type.view(out_type.shape[0], num_classes)
        out_anomaly = self.linearDect(out)
        out_anomaly = out_anomaly.view(out_anomaly.shape[0], 1, 2)
        out_anomaly = self.normDect(out_anomaly)
        out_anomaly = out_anomaly.view(out_anomaly.shape[0], 2)
        return out_type, out_anomaly

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainPath', '-train',  type=str, default="")
    parser.add_argument('--testPath', '-test', type=str, default="")
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1)
    parser.add_argument('--name', '-n', type=str, default="")
    
    config = parser.parse_args()
    trainPath = config.trainPath
    testPath = config.testPath
    learning_rate = config.learning_rate
    name = config.name
    # print(trainPath, testPath, learning_rate)
    trainData = TrainData(trainPath)
    trainLoader = torch.utils.data.DataLoader(trainData, batch_size, shuffle=True)

    testData = TestData(testPath)
    testLoader = torch.utils.data.DataLoader(testData, 1, drop_last=True)

    model = LSTM()
    # model.cuda()

    # Loss and Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    criterion = nn.CrossEntropyLoss()

    classAcurList = []
    faultAcurList = []
    totalAcurList = []

    # Train the Model
    total_step = len(trainLoader)
    for epoch in range(num_epochs):
        for i, (datas, labels, blabels, files) in enumerate(trainLoader):            
            # print(datas.shape[0], datas.shape[1], datas.shape[2], datas.shape[2])
            # print(files)
            # print(labels)
            # Forward + Backward + Optimize
            out_diog, out_dect = model(datas)
            #print(outputs)
            loss1 = criterion(out_diog, labels)
            loss2 = criterion(out_dect, blabels)
            optimizer.zero_grad()
            loss = loss1 + loss2
            # loss = loss1
            # print(loss1, loss2)
            loss.backward()
            optimizer.step()
            #print(datas, labels, files)
            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.8f'
                %(epoch+1, num_epochs, i + 1, total_step, loss.item()))
        total_correct = 0
        total_correct_dect = 0
        correct = 0
        correct_dect = 0
        correctFault = 0
        totaltotal = 0
        total = 0
        correctLabel = list(0. for i in range(num_classes))
        totalLabel = list(0. for i in range(num_classes))    
        wronglabel = list(list(0. for _ in range(num_classes)) for _ in range(num_classes))
        # 各个agent结果融合的算法
        matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        for item in testLoader:
            total_output = []
            total_detect = []
            total_pred = []
            total_process = []
            total_label = -1
            total_blabels = -1
            totaltotal += 1
            for datas, labels, blabels, files in item:
                # datas = datas.squeeze()
                # print(datas[0])
                outputs, out_dect = model(datas)
                _, predicted = torch.max(outputs.data, 1)
                total_pred.append(predicted[0].item())
                _, pred_dect = torch.max(out_dect.data, 1)
                # print("before", predicted[0].item(), total_output, outputs.data)
                total_process.append(outputs.data)
                if total_output == []:
                    total_output = outputs
                else:
                    total_output += outputs.data
                if total_detect == []:
                    total_detect = out_dect
                elif pred_dect[0].item() != 0:
                    total_detect = out_dect
                # print("after", predicted[0].item(), total_output, outputs.data)
                if total_label == -1:
                    total_label = labels
                if total_blabels == -1:
                    total_blabels = blabels
                # res = predicted.cuda() == labels
                # res_dect = pred_dect.cuda() == blabels
                res = predicted == labels
                res_dect = pred_dect == blabels
                # print(res, res_dect)
                total += labels.size(0)
                #print(datas, labels, files)
                matrix[labels[0].item()][predicted[0].item()] += 1
                for i in range(len(labels)):
                    labelSingle = labels[i]
                    if (testData.file_label[labelSingle.item()] == "normal" or testData.file_label[predicted[0].item()] == "normal") \
                        and (testData.file_label[labelSingle.item()] != "normal" or testData.file_label[predicted[0].item()] != "normal"):
                        correctFault += 1
                # correct += (predicted.cuda() == labels).sum()
                # correct_dect += (pred_dect.cuda() == blabels).sum()
                correct += (predicted == labels).sum()
                correct_dect += (pred_dect == blabels).sum()
            _, total_out = torch.max(total_output, 1)
            _, total_dect = torch.max(total_detect, 1)
            # if epoch > 1:
            #     # print(total_out[0].item(), total_pred, total_label)
            #     if total_out[0].item() == 0:
            #         # print(total_out[0].item(), total_pred, total_label, total_process)
            # total_correct += (total_out.cuda() == total_label).sum()
            # total_correct_dect += (total_dect.cuda() == total_blabels).sum()
            total_correct += (total_out == total_label).sum()
            total_correct_dect += (total_dect == total_blabels).sum()
        print(testData.file_label, matrix)
        
        classAcur = 100 * correct / total
        totalAcur = 100 * total_correct / totaltotal
        classAcurList.append(round(classAcur.item(), 2))
        totalAcurList.append(round(totalAcur.item(), 2))
        faultAcur = 100 * (total - correctFault) / total
        faultAcurList.append(round(faultAcur, 2))
        print('Epoch [%d/%d], Step [%d/%d], Loss: %.8f, accuracy: %.2f %%, fault_correct %.2f %%, total_accuracy %.2f %%, total_detect %.2f %%'
            %(epoch + 1, num_epochs, i + 1, total_step, loss.item(), 100 * correct / total, 100 * correct_dect / total  \
                , 100 * total_correct / totaltotal, 100 * total_correct_dect / totaltotal))

    # classAcurList = pd.DataFrame(classAcurList)

    # classAcurList.to_csv(drawPath + file + '.csv', index=0)

    # faultAcurList = pd.DataFrame(faultAcurList)

    # faultAcurList.to_csv(drawPath + file + '.csv', index=0)

    with open(name+'res.txt', 'a+') as f2:
        f2.writelines("LSTM" + '\n')
        f2.writelines(trainPath + '\n')
        f2.writelines(testPath + '\n')
        f2.writelines(str(learning_rate) + '\n')
        f2.writelines(str(classAcurList) + '\n')
        f2.writelines(str(faultAcurList) + '\n')
        f2.writelines(str(totalAcurList) + '\n')
        f2.writelines(str(max(classAcurList)) + '\n')
        f2.writelines(str(max(faultAcurList)) + '\n')
        f2.writelines(str(max(totalAcurList)) + '\n')
        f2.writelines('\n')

    # Save the Model
    torch.save(model.state_dict(), name+'oldmodel.pkl')