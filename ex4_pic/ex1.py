import numpy as np
import sys
from mnist import MNIST
mndata = MNIST("/export/home/016/a0166267/le4nn/")

SIZEX = 28
SIZEY = 28

PIC = 10000

MID1 = 5
MID2 = 10

def sig(t):
    return 1/(1+np.exp((-1)*int(t)))

def sigArr(a):
    return (np.apply_along_axis(sig, 1, np.array(a)))

def sof(a):
    m = np.amax(np.array(a))
    row = (np.array(a)).shape[0]
    all_m = np.full((row,1),m)
    comArr1 = np.array(a) - all_m
    comArr2 = (np.apply_along_axis(np.exp, 1, comArr1))
    sum_arr = np.sum(comArr2)
    return comArr2 / sum_arr

i = input()
if int(i) < 0 or PIC <= int(i):
    print("error!")
    sys.exit()

X, Y= mndata.load_training()
X = np.array(X)
X = X.reshape((X.shape[0],SIZEX,SIZEY))
Y = np.array(Y)


#入力層. 28*28行列を784*1行列に変換
input = X[int(i)].reshape(SIZEX * SIZEY, 1) 


#中間層
np.random.seed(0)
w1 = np.random.normal(0.0, np.sqrt(1.0/(SIZEX * SIZEY)), (MID1, SIZEX*SIZEY)) #5*784
b1 = np.random.normal(0.0, np.sqrt(1.0/(SIZEX * SIZEY)), (MID1, 1)) #5*1

com11 = w1 @ input + b1
com12 = sigArr(com11)
y1 = np.c_[com12]


#出力層
np.random.seed(1)
w2 = np.random.normal(0.0, 1.0/(SIZEX * SIZEY), (MID2, MID1)) #10*5
b2 = np.random.normal(0.0, 1.0/(SIZEX * SIZEY), (MID2, 1)) #5*1

com21 = w2 @ y1 + b2
y2 = np.c_[com21]
output = np.argmax(sof(y2))


print(output)

