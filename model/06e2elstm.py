from modelutil.E2ELSTM import run
from config import trainConfig
math_path = trainConfig["math_path"]
ruler_path = trainConfig["ruler_path"]
learning_rate = trainConfig["learning_rate"]
batch_size = trainConfig["batch_size"]
epoch_num = trainConfig["epoch_num"]
model_path = 'out/model/E2ELSTM'
run(math_path,ruler_path,learning_rate,batch_size,epoch_num,model_path,init=True)
