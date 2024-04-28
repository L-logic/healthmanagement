nStart=1
nEnd=1

learning_rate=0.001
trainPath='faultdiagnosis/train9to12/'
testPath='faultdiagnosis/train9to12/'
name='project/out/model/cnn/'
# 1204126
for i in $(seq $nStart $nEnd)   
do  
echo '1186721'
nohup /root/anaconda3/envs/py37_ns/bin/python faultdiagnosis/dataTrainCNN.py -lr=$learning_rate -train=$trainPath -test=$testPath -n=$name > CNNTrain1.log 2>&1 &

echo 'end'
done