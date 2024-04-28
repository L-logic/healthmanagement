from modelutil.LstmModel import train,test
from config import trainConfig
math_path = trainConfig["math_path"]
ruler_path = trainConfig["ruler_path"]
learning_rate = trainConfig["learning_rate"]
batch_size = trainConfig["batch_size"]
epoch_num = trainConfig["epoch_num"]
model_path = 'out/model/LSTMRE'
def LstmTrain():
    train(math_path,ruler_path,learning_rate,batch_size,epoch_num,model_path,init=True)
def LstmTest():
    pass
LstmTrain()