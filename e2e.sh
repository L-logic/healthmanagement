nStart=1
nEnd=1

learning_rate=0.001
trainPath='data/origin/train9to12/'
testPath='data/origin/train9to12/'
name='project/out/model/e2e/' 

for i in $(seq $nStart $nEnd)   
do  
/root/anaconda3/envs/py37_ns/bin/python model/modelutil/E2E.py -lr=$learning_rate -train=$trainPath -test=$testPath -n=$name
done