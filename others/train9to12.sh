nStart=1
nEnd=1

learning_rate=0.001
# trainPath='./train9to12/'
# testPath='./val9to12/'
# trainPath='faultdiagnosis/train9to12/'

trainPath='langue/out/train/'
testPath='langue/out/train/'
name='train/newmodel/' 

for i in $(seq $nStart $nEnd)   
do  
nohup /root/anaconda3/envs/py37_ns/bin/python faultdiagnosis/dataTrain.py -lr=$learning_rate -train=$trainPath -test=$testPath -n=$name > newmodel.log 2>&1 &
done