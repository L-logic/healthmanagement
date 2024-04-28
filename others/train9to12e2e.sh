nStart=1
nEnd=1

learning_rate=0.001
trainPath='faultdiagnosis/train9to12/'
testPath='faultdiagnosis/train9to12/'
name='project/out/model/e2e/' 

for i in $(seq $nStart $nEnd)   
do  
nohup /root/anaconda3/envs/py37_ns/bin/python faultdiagnosis/dataTrainE2E.py -lr=$learning_rate -train=$trainPath -test=$testPath -n=$name  > E2ETrain.log 2>&1 &
done