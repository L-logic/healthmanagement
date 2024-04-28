import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F

# "cuda:4" if torch.cuda.is_available() else
device = torch.device( "cpu") 
# 超参数定义
batch_size = 20
# num_epochs = 2
input_dim = 22
length = 96
output_dim = 7
# transformer层数
num_layers = 2
dropout = 0.3

# 读入数据集
# 同时进行多分类和二分类的标签划分
class FaultDataset(Dataset):
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
        # combined_data = np.vstack([math, ruler])  
        data = torch.tensor(math)
        label = torch.tensor(label, dtype=torch.int64)
        binary_label = torch.tensor(binary_label, dtype=torch.float32)
        bn = torch.nn.BatchNorm1d(input_dim, affine=False)
        data = bn(data)
        return data, label, binary_label


# 搭建CNN+Transformer神经网络
class OneDCNNClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, dropout):
        super(OneDCNNClassifier, self).__init__()
        self.bn = torch.nn.BatchNorm1d(input_dim, affine=False)
        self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128 * length, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.fc4 = nn.Linear(128, 1)  # binary classification layer
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=16, dim_feedforward=512, dropout=dropout),
            num_layers=num_layers
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # 调整形状
        x = x.permute(2, 0, 1)
        # 使用transformer层
        x = self.transformer(x)
        # 调整形状
        x = x.permute(1, 0, 2)  
        x = x.reshape(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        multiclass_output = self.fc3(x)
        binary_output = torch.sigmoid(self.fc4(x))  # for binary classification
        return multiclass_output, binary_output



# 用于绘图
# train_losses = []
# test_losses = []
# train_accuracies_multiclass = []
# test_accuracies_multiclass = []
# train_accuracies_binary = []
# test_accuracies_binary = []

def test(testPath,ruler_path,model_path,result):  
    test_dataset = FaultDataset(testPath,ruler_path)  
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)  
    # device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')  
      
    model = OneDCNNClassifier(input_dim, output_dim, num_layers, dropout).to(device)  
    model_state_dict = torch.load(model_path, map_location=device)  
    model.load_state_dict(model_state_dict)  
    model.eval()  
    totaltotal = 0
    outputslist = []
    outputslabels = []
    out_dectlist = []
    out_dectlabel = []
    # 打开或创建csv文件  
    for i, (inputs, labels, binary_labels) in enumerate(test_loader):  
        if totaltotal==1000:
            break
        inputs, labels, binary_labels = inputs.to(device), labels.to(device), binary_labels.to(device)  
        outputs, out_dect = model(inputs)          
        # loss1 = criterion(outputs, labels)
        # loss2 = criterion(out_dect, blabels)
        _, predicted_multiclass = torch.max(outputs.data, 1)
        predicted_binary = (out_dect > 0.5).view(-1).long()
        outputslist.extend(predicted_multiclass.cpu().numpy().tolist())
        outputslabels.extend(labels.cpu().numpy().tolist())
        out_dectlist.extend(predicted_binary.cpu().numpy().tolist())
        out_dectlabel.extend(binary_labels.cpu().numpy().tolist())
        totaltotal += 1
    df = pd.DataFrame()  
    df['outputslist'] = outputslist
    df['outputslabels'] = outputslabels
    df['out_dectlist'] = out_dectlist
    df['out_dectlabel'] = out_dectlabel
    df.to_csv(result, index=False)  
    print('result',result)
def train(trainPath,testPath,rulerpath,learning_rate,name,num_epochs):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--trainPath', '-train',  type=str, default="")
    # parser.add_argument('--testPath', '-test', type=str, default="")
    # parser.add_argument('--learning_rate', '-lr', type=float, default=0.1)
    # parser.add_argument('--name', '-n', type=str, default="")

    # config = parser.parse_args()
    # trainPath = config.trainPath
    # testPath = config.testPath
    # name = config.name


    # 测试集、训练集等
    train_dataset = FaultDataset(trainPath,rulerpath)
    test_dataset = FaultDataset(trainPath,rulerpath)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    # 定义设备
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    # 定义模型
    model = OneDCNNClassifier(input_dim, output_dim, num_layers, dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    # 训练测试阶段，使用多任务学习
    train_losses = []
    train_accuracies_multiclass=[]
    train_accuracies_binary = []
    with open(name+'training_results.txt', 'a+') as f:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            correct_multiclass = 0
            correct_binary = 0
            total = 0
            for i, (inputs, labels, binary_labels) in enumerate(train_loader):
                inputs, labels, binary_labels = inputs.to(device), labels.to(device), binary_labels.to(device)
                multiclass_outputs, binary_outputs = model(inputs)
                loss_multiclass = criterion(multiclass_outputs, labels)
                loss_binary = F.binary_cross_entropy(binary_outputs.view(-1), binary_labels)
                loss = loss_multiclass + loss_binary
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _, predicted_multiclass = torch.max(multiclass_outputs.data, 1)
                predicted_binary = (binary_outputs > 0.5).view(-1).long()
                total += labels.size(0)
                correct_multiclass += (predicted_multiclass == labels).sum().item()
                correct_binary += (predicted_binary == binary_labels).sum().item()
            train_loss = train_loss / len(train_loader)
            train_accuracies_multiclass.append(100 * correct_multiclass / total)
            train_accuracies_binary.append(100 * correct_binary / total)
            f.write(f'{epoch + 1},{loss_multiclass:.4f},{loss_binary:.4f},{train_loss:.4f}, {train_accuracies_multiclass[-1]:.2f}%, {train_accuracies_binary[-1]:.2f}%\n')  
            
            torch.save(model.state_dict(),name+ f'model{epoch + 1}.pkl')
            # 训练完成后，文件将自动关闭
            # print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy Multiclass: {train_accuracies_multiclass[-1]:.2f}%, Train Accuracy Binary: {train_accuracies_binary[-1]:.2f}%')
            
    #         model.eval()
    #         test_loss = 0
    #         correct_multiclass = 0
    #         correct_binary = 0
    #         total = 0
    #         with torch.no_grad():
    #             for inputs, labels, binary_labels in test_loader:
    #                 inputs, labels, binary_labels = inputs.to(device), labels.to(device), binary_labels.to(device)
    #                 multiclass_outputs, binary_outputs = model(inputs)
    #                 loss_multiclass = criterion(multiclass_outputs, labels)
    #                 loss_binary = F.binary_cross_entropy(binary_outputs.view(-1), binary_labels)
    #                 loss = loss_multiclass + loss_binary
    #                 test_loss += loss.item()
    #                 _, predicted_multiclass = torch.max(multiclass_outputs.data, 1)
    #                 predicted_binary = (binary_outputs > 0.5).view(-1).long()
    #                 total += labels.size(0)
    #                 correct_multiclass += (predicted_multiclass == labels).sum().item()
    #                 correct_binary += (predicted_binary == binary_labels).sum().item()
    #         test_loss = test_loss / len(test_loader)
    #         test_losses.append(test_loss)
    #         test_accuracies_multiclass.append(round(100 * correct_multiclass / total, 2))
    #         test_accuracies_binary.append(round(100 * correct_binary / total, 2))
    #         print(
    #             f'Epoch {epoch + 1}, Test Loss: {test_loss:.4f}, Test Accuracy Multiclass: {test_accuracies_multiclass[-1]:.2f}%, Test Accuracy Binary: {test_accuracies_binary[-1]:.2f}%')
    #         optimizer.step()

    # with open(name+"result.txt", 'a+') as f2:
    #     f2.writelines("LSTMTrans" + '\n')
    #     f2.writelines(trainPath + '\n')
    #     f2.writelines(testPath + '\n')
    #     f2.writelines(str(learning_rate) + '\n')
    #     f2.writelines(str(test_accuracies_multiclass) + '\n')
    #     f2.writelines(str(test_accuracies_binary) + '\n')
    #     f2.writelines(str(max(test_accuracies_multiclass)) + '\n')
    #     f2.writelines(str(max(test_accuracies_binary)) + '\n')
    #     f2.writelines('\n')
    # Save the Model
    torch.save(model.state_dict(),name+ 'model.pkl')

# # 绘制图像（可删除）
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(train_accuracies_multiclass, 'b-', label='Train Accuracy Multiclass')
# plt.plot(test_accuracies_multiclass, 'r-', label='Test Accuracy Multiclass')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Multiclass Accuracy')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(train_accuracies_binary, 'b-', label='Train Accuracy Binary')
# plt.plot(test_accuracies_binary, 'r-', label='Test Accuracy Binary')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Binary Accuracy')
# plt.legend()

# plt.tight_layout()
# plt.show()
