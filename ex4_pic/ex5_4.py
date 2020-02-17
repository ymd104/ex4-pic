# Adam

import numpy as np
import numpy.matlib
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm
# from numpy.core.multiarray import ndarray

mndata = MNIST("/export/home/016/a0166267/le4nn/")

SIZEX = 28
SIZEY = 28
PIC = 50000
MID1 = 500
MID2 = 10

BATCH_SIZE = 100
EPOCH = 1

ALPHA = 0.001
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8

X, Y = mndata.load_training()
X = np.array(X)
X = X.reshape((X.shape[0], SIZEX, SIZEY))
Y = np.array(Y)


def sig(p):
    sig_range = 34.538776394910684

    if p <= -sig_range:
        return 1e-15
    if p >= sig_range:
        return 1.0 - 1e-15

    return 1 / (1 + np.exp(-p))


def sigarr(a):
    return np.vectorize(sig)(a)


def sigd(p):
    return (1 - sig(p)) * sig(p)


def sigdarr(a):
    return np.vectorize(sigd)(a)


def sof(a):
    m = np.amax(np.array(a))
    return(np.exp(a - m))/np.sum(np.exp(a - m))


def rowsum(a):
    comsum = np.sum(a, axis=1)
    return np.c_[comsum]


def crossentropy(p, y):
    delta = 1e-7  # 微小な値
    return -np.sum(np.dot(p, np.log(y + delta)))


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
    t = 0
    mw1 = 0
    vw1 = 0
    mw2 = 0
    vw2 = 0
    mb1 = 0
    vb1 = 0
    mb2 = 0
    vb2 = 0
else:
    w1 = np.load('ex54_w1.npy')
    b1 = np.load('ex54_b1.npy')
    w2 = np.load('ex54_w2.npy')
    b2 = np.load('ex54_b2.npy')
    t = np.load('ex54_t.npy')
    mw1 = np.load('ex54_mw1.npy')
    vw1 = np.load('ex54_vw1.npy')
    mw2 = np.load('ex54_mw2.npy')
    vw2 = np.load('ex54_vw2.npy')
    mb1 = np.load('ex54_mb1.npy')
    vb1 = np.load('ex54_vb1.npy')
    mb2 = np.load('ex54_mb2.npy')
    vb2 = np.load('ex54_vb2.npy')

print("入力待ち：0で学習モード, 1で学習済みデータを利用した画像識別")
mode = int(input())
if mode == 0:

    repeat = int(np.floor(PIC / BATCH_SIZE * EPOCH))
    graph_ent_error = []

    for i in range(repeat):

        num_arr = np.random.randint(0, PIC - 1, BATCH_SIZE)

        entropy_average = 0

        # 学習開始.
        en_Ak = np.empty((0, MID2))  # dEn/dAk. B*C行列.
        x_ = np.empty((MID1, 0))  # X for phase5.

        x_input = np.apply_along_axis(picx, 0, num_arr)
        x_input = x_input.reshape((BATCH_SIZE, SIZEX * SIZEY))/255

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

        t += 1

        mw1 = BETA1 * mw1 + (1 - BETA1) * en_w_1
        vw1 = BETA2 * vw1 + (1 - BETA2) * (en_w_1 * en_w_1)
        m_tmp_w1 = mw1 / (1 - np.power(BETA1, t))
        v_tmp_w1 = vw1 / (1 - np.power(BETA2, t))
        w1 = w1 - (ALPHA * m_tmp_w1) / (np.sqrt(v_tmp_w1) + EPSILON)

        mw2 = BETA1 * mw2 + (1 - BETA1) * en_w_2
        vw2 = BETA2 * vw2 + (1 - BETA2) * (en_w_2 * en_w_2)
        m_tmp_w2 = mw2 / (1 - np.power(BETA1, t))
        v_tmp_w2 = vw2 / (1 - np.power(BETA2, t))
        w2 = w2 - (ALPHA * m_tmp_w2) / (np.sqrt(v_tmp_w2) + EPSILON)

        mb1 = BETA1 * mb1 + (1 - BETA1) * en_b_1
        vb1 = BETA2 * vb1 + (1 - BETA2) * (en_b_1 * en_b_1)
        m_tmp_b1 = mb1 / (1 - np.power(BETA1, t))
        v_tmp_b1 = vb1 / (1 - np.power(BETA2, t))
        b1 = b1 - (ALPHA * m_tmp_b1) / (np.sqrt(v_tmp_b1) + EPSILON)

        mb2 = BETA1 * mb2 + (1 - BETA1) * en_b_2
        vb2 = BETA2 * vb2 + (1 - BETA2) * (en_b_2 * en_b_2)
        m_tmp_b2 = mb2 / (1 - np.power(BETA1, t))
        v_tmp_b2 = vb2 / (1 - np.power(BETA2, t))
        b2 = b2 - (ALPHA * m_tmp_b2) / (np.sqrt(v_tmp_b2) + EPSILON)

        print(i)
        print(entropy_average)
        graph_ent_error.append(entropy_average)


        # 学習終了#

    num = list(range(repeat))
    plt.plot(num, graph_ent_error, label = "error")
    plt.show()

    # 精度の計測

    np.save('ex54_w1.npy', w1)
    np.save('ex54_w2.npy', w2)
    np.save('ex54_b1.npy', b1)
    np.save('ex54_b2.npy', b2)
    np.save('ex54_t.npy', t)
    np.save('ex54_mw1.npy', mw1)
    np.save('ex54_mw2.npy', mw2)
    np.save('ex54_vw1.npy', vw1)
    np.save('ex54_vw2.npy', vw2)
    np.save('ex54_mb1.npy', mb1)
    np.save('ex54_mb2.npy', mb2)
    np.save('ex54_vb1.npy', vb1)
    np.save('ex54_vb2.npy', vb2)
    right = 0
    wrong = 0

    #"""
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
    #"""

print("入力画像のIDを入力してください")
id_pic = int(input())

#入力層. 28*28行列を784*1行列に変換
input_pic = X[id_pic].reshape(SIZEX * SIZEY, 1)

#中間層

y1 = np.c_[sigarr(w1 @ input_pic + b1)]

#出力層

com21 = w2 @ y1 + b2
y2 = np.c_[com21]
sofy2 = sof(y2)
output = np.argmax(sofy2)

print(output)
plt.imshow(X[id_pic], cmap=cm.gray)
plt.show()