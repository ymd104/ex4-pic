# AdaGrad

import numpy as np
import numpy.matlib
from mnist import MNIST
# from numpy.core.multiarray import ndarray

mndata = MNIST("/export/home/016/a0166267/le4nn/")

SIZEX = 28
SIZEY = 28
PIC = 50000
MID1 = 500
MID2 = 10

BATCH_SIZE = 100

LEARNING_RATE = 0.001
h_initialized = 1e-8

X, Y = mndata.load_training()
X = np.array(X)
X = X.reshape((X.shape[0], SIZEX, SIZEY))
Y = np.array(Y)


def sig(t):
    sig_range = 34.538776394910684

    if t <= -sig_range:
        return 1e-15
    if t >= sig_range:
        return 1.0 - 1e-15

    return 1 / (1 + np.exp(-t))


def sigarr(a):
    return np.vectorize(sig)(a)


def sigd(t):
    return (1 - sig(t)) * sig(t)


def sigdarr(a):
    return np.vectorize(sigd)(a)


def sof(a):
    m = np.amax(np.array(a))
    return(np.exp(a - m))/np.sum(np.exp(a - m))


def rowsum(a):
    comsum = np.sum(a, axis=1)
    return np.c_[comsum]


def crossentropy(t, y):
    delta = 1e-7  # 微小な値
    return -np.sum(np.dot(t, np.log(y + delta)))


def picx(m):
    return X[m]


print("学習パラメータを初期化しますか？")
print("(1で初期化)")
initialize = input()
if initialize == "1":
    np.random.seed(0)
    w1 = np.random.normal(0.0, np.sqrt(1.0/(SIZEX * SIZEY)), (MID1, SIZEX*SIZEY))  # mid2*784
    b1 = np.random.normal(0.0, np.sqrt(1.0/(SIZEX * SIZEY)), (MID1, 1))  # mid2*1
    w2 = np.random.normal(0.0, np.sqrt(1.0/MID1), (MID2, MID1))  # 10*mid2
    b2 = np.random.normal(0.0, np.sqrt(1.0/MID1), (MID2, 1))  # 10*1
    h1 = h_initialized
    h2 = h_initialized
else:
    w1 = np.load('ex52_w1.npy')
    b1 = np.load('ex52_b1.npy')
    w2 = np.load('ex52_w2.npy')
    b2 = np.load('ex52_b2.npy')
    h1 = np.load('ex52_h1.npy')
    h2 = np.load('ex52_h2.npy')


for i in range(500):

    num_arr = np.random.randint(0, PIC - 1, BATCH_SIZE)

    entropy_average = 0

    # 学習開始.
    en_Ak = np.empty((0, MID2))  # dEn/dAk. B*C行列.
    x_ = np.empty((MID1, 0))  # X for phase5.

    x_input = np.apply_along_axis(picx, 0, num_arr)
    x_input = x_input.reshape((BATCH_SIZE, SIZEX * SIZEY))

    # 中間層

    y1 = np.c_[sigarr(w1 @ x_input.T + np.matlib.repmat(b1, 1, BATCH_SIZE))]

    com21 = w2 @ y1 + np.matlib.repmat(b2, 1, BATCH_SIZE)
    y2 = np.c_[com21]
    sofy2 = np.apply_along_axis(sof, 0, y2)
    output = np.argmax(sofy2)

    k = np.zeros((BATCH_SIZE, MID2))
    count = 0
    for n in num_arr:
        k[count, Y[n]] = 1
        count += 1
    entropy_average = (-1) * np.sum(k.T * np.log(sofy2)) / BATCH_SIZE

    #--- 学習 ---#

    # phase4#

    en_ak = np.array((sofy2.T - k) / BATCH_SIZE)

    # phase5#

    en_x_2 = np.dot(w2.T, en_ak.T)
    en_w_2 = np.dot(en_ak.T, y1.T)
    en_b_2 = rowsum(en_ak.T)

    # phase6#

    en_y_1 = en_x_2 * sigdarr(en_x_2)

    # phase7#

    en_x_1 = np.dot(w1.T, en_y_1)
    en_w_1 = np.dot(en_y_1, x_input)
    en_b_1 = rowsum(en_y_1)

    # phase8#

    h1 = h1 + en_w_1 * en_w_1
    tmp1 = 1 / np.sqrt(h1)
    w1 = w1 - LEARNING_RATE * (tmp1 * en_w_1)
    h2 = h2 + en_w_2 * en_w_2
    tmp2 = 1 / np.sqrt(h2)
    w2 = w2 - LEARNING_RATE * (tmp2 * en_w_2)
    b1 = b1 - LEARNING_RATE * en_b_1
    b2 = b2 - LEARNING_RATE * en_b_2

    print(i)
    print(entropy_average)

    # 学習終了#


# 精度の計測

print(w1)
print(b1)
print(w2)
print(b2)
np.save('ex52_w1.npy', w1)
np.save('ex52_w2.npy', w2)
np.save('ex52_b1.npy', b1)
np.save('ex52_b2.npy', b2)
np.save('ex52_h1.npy', h1)
np.save('ex52_h2.npy', h2)

right = 0
wrong = 0

for j in range(10000):

    j += 50000

#入力層. 28*28行列を784*1行列に変換
    input_pic = X[j].reshape(SIZEX * SIZEY, 1)


#中間層

    y1 = np.c_[sigarr(w1 @ input_pic + b1)]

#出力層

    com21 = w2 @ y1 + b2
    y2 = np.c_[com21]
    sofy2 = sof(y2)
    output = np.argmax(sofy2)

    k = Y[j]
    if k == output:
        right += 1
    else:
        wrong += 1

print(right/(right + wrong))
