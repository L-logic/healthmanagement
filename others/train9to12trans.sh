nStart=1
nEnd=1

learning_rate=0.001
trainPath='faultdiagnosis/train9to12/'
testPath='faultdiagnosis/train9to12/'
name='result9to12e2etrans' 

for i in $(seq $nStart $nEnd)   
do  
/root/anaconda3/envs/py37_ns/bin/python /root/public/wxn/ns3-demo/faultdiagnosis/dataTrans.py -lr=$learning_rate -train=$trainPath -test=$testPath -n=$name
done