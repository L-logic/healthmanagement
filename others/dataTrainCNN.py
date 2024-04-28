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
learning_rate = 0.001
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu") 
# Hyper Parameters
num_e2e = 3
input_size = 18  # 输入数据的维度
length = 96  # 输入数据的长度
num_classes = 7
batch_size = 40 
num_epochs = 40

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
        f = open(self.data_path + '/' + file.split('_')[0] + '/' + file, 'r+')
        csv_reader = csv.reader(f)
        sample = []
        # for line in csv_reader:
        #     row = list(map(float, line))
        #     sample.append(row)
        for _ in range(num_e2e):
            sample.append([])
        for line in csv_reader:
            row = list(map(float, line))
            e2e0 = row[0:5]
            e2e1 = row[0:5]
            e2e2 = row[0:5]
            for i in range(5, 11):
                e2e0.append(row[i])
            e2e0.append(0)
            for i in range(11, 17):
                e2e1.append(row[i])
            e2e1.append(0)
            for i in range(16, 22):
                e2e2.append(row[i])
            e2e2.append(0)
            sample[0].append(e2e0)
            sample[1].append(e2e1)
            sample[2].append(e2e2)
        f.close()
        sample = torch.tensor(sample, dtype=torch.float32).to(device)
        label = torch.tensor(label, dtype=torch.int64).to(device)
        blabel = torch.tensor(blabel, dtype=torch.int64).to(device)
        
        # mean_a = torch.mean(data, dim=1)
        # std_a = torch.std(data, dim=1)

        # # Do Z-score standardization on 2D tensor
        # n_a = data.sub_(mean_a[:, None]).div_(std_a[:, None])

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
        # print(self.datas[index])
        for file, label, blabel in self.datas[index]:
            f = open(self.data_path + '/' + file.split('_')[0] + '/' + file, 'r+')
            csv_reader = csv.reader(f)
            sample = []
            for _ in range(num_e2e):
                sample.append([])
            for line in csv_reader:
                row = list(map(float, line))
                e2e0 = row[0:5]
                e2e1 = row[0:5]
                e2e2 = row[0:5]
                for i in range(5, 11):
                    e2e0.append(row[i])
                e2e0.append(0)
                for i in range(11, 17):
                    e2e1.append(row[i])
                e2e1.append(0)
                for i in range(16, 22):
                    e2e2.append(row[i])
                e2e2.append(0)
                sample[0].append(e2e0)
                sample[1].append(e2e1)
                sample[2].append(e2e2)
            f.close()

            sample = torch.tensor(sample, dtype=torch.float32).to(device)
            label = torch.tensor(label, dtype=torch.int64).to(device)
            blabel = torch.tensor(blabel, dtype=torch.int64).to(device)
            item.append((sample, label, blabel, file))
        return item

class LSTM(nn.Module):
    def __init__(self):
        # 50 200 0.0002 2 69.31 85.03

        self.learning_rate = learning_rate
        super(LSTM, self).__init__()
        self.num_layers = 2
        self.conv_size = 64
        # self.conv_size = 48        
        self.hidden_size = 48
        # self.linear1 = nn.Linear(input_size, 16)
        self.conv = nn.Sequential(
            nn.Conv2d(num_e2e, self.conv_size, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(self.conv_size),
            nn.Conv2d(self.conv_size, self.hidden_size, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(self.hidden_size),
            )
        self.linear1 = nn.Linear(72* self.hidden_size, int(self.hidden_size / 2))
        self.linear2 = nn.Linear(int(self.hidden_size / 2), int(self.hidden_size / 4))
        self.linearDiog = nn.Linear(int(self.hidden_size / 4), num_classes)
        self.linearDect = nn.Linear(int(self.hidden_size / 4), 2)
        self.normDiog = nn.BatchNorm1d(1)
        self.normDect = nn.BatchNorm1d(1)

    def forward(self, x):
        # # 设置初始状态h_0与c_0的状态是初始的状态，一般设置为0，尺寸是,x.size(0)
        # print("?????", x.shape)
        # x = self.spacial_attention(x) * x
        # new = torch.zeros(x.shape[0], x.shape[2], x.shape[3] * 3).to(device)
        # for i in range(x.shape[0]):
        #     e1 = x[i][0]
        #     e2 = x[i][1]
        #     e3 = x[i][2]
        #     new[i] = torch.cat([e1, e2, e3], 1)
        x = self.conv(x)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        out = self.linear1(x)
        out = self.linear2(out)             # 经过线性层后，out的shape为(batch_size, n_class)
        out_type = self.linearDiog(out)
        # print(out_type.shape)
        out_type = out_type.view(out_type.shape[0], 1, num_classes)
        out_type = self.normDiog(out_type)
        out_type = out_type.view(out_type.shape[0], num_classes)
        out_anomaly = self.linearDect(out)
        out_anomaly = out_anomaly.view(out_anomaly.shape[0], 1, 2)
        out_anomaly = self.normDect(out_anomaly)
        out_anomaly = out_anomaly.view(out_anomaly.shape[0], 2)
        return out_type, out_anomaly

def test(name,result_path):
    testPath = 'faultdiagnosis/val9to12/'
    testData = TestData(testPath)
    testLoader = torch.utils.data.DataLoader(testData, 1, drop_last=True)

    model = LSTM()
    model.to(device)
    model_state_dict = torch.load(name , map_location=device)  
    model.load_state_dict(model_state_dict)  
    model.eval()  
    # 打开或创建csv文件  
    totaltotal = 0
    outputslist = []
    outputslabels = []
    out_dectlist = []
    out_dectlabel = []
    for item in testLoader:
        for datas, labels, blabels, files in item:
            outputs, out_dect = model(datas)            
            # loss1 = criterion(outputs, labels)
            # loss2 = criterion(out_dect, blabels)
            _, pred_labels = torch.max(outputs, 1)   
            outputslist.extend(pred_labels.cpu().numpy().tolist())
            outputslabels.extend(labels.cpu().numpy().tolist())
            _, pred_dect = torch.max(out_dect, 1)   
            out_dectlist.extend(pred_dect.cpu().numpy().tolist())
            out_dectlabel.extend(blabels.cpu().numpy().tolist())
            totaltotal += 1
    df = pd.DataFrame()  
    df['outputslist'] = [o for o in outputslist] 
    df['outputslabels'] =  [out for out in outputslabels]
    df['out_dectlist'] = [d for d in out_dectlist]
    df['out_dectlabel'] = [out for out in out_dectlabel]
    # csv_filename = 'project/img/max-data.csv'  
    df.to_csv(result_path, index=False)  
    
    print(f"数据已保存到 {result_path} 文件中。")
# def train(math_path,ruler_path,testPath,name,learning_rate,num_epochs,model_path='')
# if __name__=='__main__':
#     parser = argparse.ArgumentParser()
#     # parser.add_argument('--trainPath', '-train',  type=str, default="")
#     # parser.add_argument('--testPath', '-test', type=str, default="")
#     # parser.add_argument('--learning_rate', '-lr', type=float, default=0.1)
#     # parser.add_argument('--name', '-n', type=str, default="")
#     # trainPath="project/data/math/"
#     trainPath="faultdiagnosis/train9to12/"
#     testPath="faultdiagnosis/train9to12/"
#     # config = parser.parse_args()
#     # trainPath = config.trainPath
#     # testPath = config.testPath
#     # learning_rate = config.learning_rate
#     # name = config.name
#     # print(trainPath, testPath, learning_rate)
#     trainData = TrainData(trainPath)
#     trainLoader = torch.utils.data.DataLoader(trainData, batch_size, shuffle=True)

#     testData = TestData(testPath)
#     testLoader = torch.utils.data.DataLoader(testData, 1, drop_last=True)

#     model = LSTM()
#     model.to(device)

#     # Loss and Optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
#     criterion = nn.CrossEntropyLoss()

#     classAcurList = []
#     faultAcurList = []
#     totalAcurList = []

#     # Train the Model
#     total_step = len(trainLoader)
#     for epoch in range(num_epochs):
#         for i, (datas, labels, blabels, files) in enumerate(trainLoader):            
#             # print(datas.shape[0], datas.shape[1], datas.shape[2], datas.shape[2])
#             # print(files)
#             #print(labels)
#             # Forward + Backward + Optimize
#             # print(datas.shape)
#             out_diog, out_dect = model(datas)
#             #print(outputs)

#             loss1 = criterion(out_diog, labels)
#             loss2 = criterion(out_dect, blabels)
#             optimizer.zero_grad()
#             loss = loss1 + loss2
#             # loss = loss1
#             # print(loss1, loss2)
#             loss.backward()
#             optimizer.step()
#             #print(datas, labels, files)
#             if (i + 1) % 100 == 0:
#                 print('Epoch [%d/%d], Step [%d/%d], Loss: %.8f'
#                 %(epoch+1, num_epochs, i + 1, total_step, loss.item()))
#         torch.save(model.state_dict(), f'project/out/model/cnn1/model{epoch+1}.pkl')
#         # total_correct = 0
#         # total_correct_dect = 0
#         # correct = 0
#         # correct_dect = 0
#         # correctFault = 0
#         # totaltotal = 0
#         # total = 0
#         # correctLabel = list(0. for i in range(num_classes))
#         # totalLabel = list(0. for i in range(num_classes))    
#         # wronglabel = list(list(0. for _ in range(num_classes)) for _ in range(num_classes))
#         # # 各个agent结果融合的算法
#         # matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
#         # matrix_total = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
#         # for item in testLoader:
#         #     # 单次数字孪生实验内
#         #     total_output = []
#         #     total_detect = []
#         #     total_pred = []
#         #     total_process = []
#         #     total_label = -1
#         #     total_blabels = -1
#         #     totaltotal += 1
#         #     for datas, labels, blabels, files in item:
#         #         # datas = datas.squeeze()
#         #         # print(datas[0])
#         #         outputs, out_dect = model(datas)
#         #         _, predicted = torch.max(outputs.data, 1)
#         #         total_pred.append(predicted[0].item())
#         #         _, pred_dect = torch.max(out_dect.data, 1)
#         #         # print("before", predicted[0].item(), total_output, outputs.data)
#         #         total_process.append(outputs.data)
#         #         if total_output == []:
#         #             total_output = outputs.data
#         #         else:
#         #             total_output += outputs.data
#         #         if total_detect == []:
#         #             total_detect = out_dect
#         #         elif pred_dect[0].item() != 0:
#         #             total_detect = out_dect
#         #         # print("after", predicted[0].item(), total_output, outputs.data)
#         #         if total_label == -1:
#         #             total_label = labels
#         #         if total_blabels == -1:
#         #             total_blabels = blabels
#         #         res = predicted.to(device) == labels
#         #         res_dect = pred_dect.to(device) == blabels
#         #         # print(res, res_dect)
#         #         total += labels.size(0)
#         #         #print(datas, labels, files)
#         #         matrix[labels[0].item()][predicted[0].item()] += 1
#         #         for i in range(len(labels)):
#         #             labelSingle = labels[i]
#         #             if (testData.file_label[labelSingle.item()] == "normal" or testData.file_label[predicted[0].item()] == "normal") \
#         #                 and (testData.file_label[labelSingle.item()] != "normal" or testData.file_label[predicted[0].item()] != "normal"):
#         #                 correctFault += 1
#         #             correctLabel[labelSingle] += res[i].item()
#         #             totalLabel[labelSingle] += 1
#         #         correct += (predicted.to(device) == labels).sum()
#         #         correct_dect += (pred_dect.to(device) == blabels).sum()
#         #     _, total_out = torch.max(total_output, 1)
#         #     _, total_dect = torch.max(total_detect, 1)
#         #     # if epoch > 1:
#         #     #     # print(total_out[0].item(), total_pred, total_label)
#         #     #     if total_out[0].item() == 0:
#         #     #         # print(total_out[0].item(), total_pred, total_label, total_process)
#         #     matrix_total[total_label[0].item()][total_out[0].item()] += 1
#         #     total_correct += (total_out.to(device) == total_label).sum()
#         #     total_correct_dect += (total_dect.to(device) == total_blabels).sum()
#         # print(testData.file_label, matrix, matrix_total)
        
#         # classAcur = 100 * correct / total
#         # totalAcur = 100 * total_correct / totaltotal
#         # classAcurList.append(round(classAcur.item(), 2))
#         # totalAcurList.append(round(totalAcur.item(), 2))
#         # faultAcur = 100 * (total - correctFault) / total
#         # faultAcurList.append(round(faultAcur, 2))
#         # print('Epoch [%d/%d], Step [%d/%d], Loss: %.8f, accuracy: %.2f %%, fault_correct %.2f %%, total_accuracy %.2f %%, total_detect %.2f %%'
#         #     %(epoch + 1, num_epochs, i + 1, total_step, loss.item(), 100 * correct / total, 100 * correct_dect / total  \
#         #         , 100 * total_correct / totaltotal, 100 * total_correct_dect / totaltotal))

#     # classAcurList = pd.DataFrame(classAcurList)

#     # classAcurList.to_csv(drawPath + file + '.csv', index=0)

#     # faultAcurList = pd.DataFrame(faultAcurList)

#     # faultAcurList.to_csv(drawPath + file + '.csv', index=0)



#     # # Save the Model


epoch = 40
for index in range(epoch):
    name ="project/out/model/cnn1/"
    result_path = 'project/out/model/result/cnni1/result'
    test(f"{name}model{index+1}.pkl",f"{result_path}{index+1}.csv")