import numpy as np
def initm(tens, kernel, s, i, j):
    kw = kernel.shape[1]
    h = tens.shape[1]
    oh = int((h-kw)/s) + 1
    w = tens.shape[2]
    ow = int((w-kw)/s) + 1
    kerm = np.zeros((oh*ow, h*w))
    for k in range(oh):
        lm = k*w
        for l in range(ow):
            for n in range(kw):
                for p in range(kw):
                    kerm[k*ow + l][lm + l*s + n*w + p] = kernel[n][p][i][j]
    return kerm
def initv(tens, i, j):
    h = tens.shape[1]
    w = tens.shape[2]
    tv = np.zeros(h*w)
    for k in range(h):
        for l in range(w):
            tv[k*w + l] = tens[i][k][l][j]
    return tv
def initb(tens, kernel, bias, s, j):
    kw = kernel.shape[1]
    h = tens.shape[1]
    oh = int((h-kw)/s) + 1
    w = tens.shape[2]
    ow = int((w-kw)/s) + 1
    bv = np.ones(oh*ow)
    bv = bias[j]*bv
    return bv
def conv(tens, kernel, bias, s):
    kw = kernel.shape[1]
    h = tens.shape[1]
    oh = int((h-kw)/s) + 1
    w = tens.shape[2]
    ow = int((w-kw)/s) + 1
    batch = tens.shape[0]
    cout = kernel.shape[3]
    cin = tens.shape[3]
    p = list(range(batch))
    for b in range(batch):
        p[b] = []
        for j in range(cout):
            res = np.zeros(oh*ow)
            for i in range(cin):
                kerm = initm(tens, kernel, s, i, j)
                tv = initv(tens, b, i)
                r = kerm @ tv
                res = res + r
            res = res + initb(tens, kernel, bias, s, j)
            res = np.reshape(res,(oh,-1))
            p[b].append(res)
    return np.transpose(np.array(p), (0, 2, 3, 1))

with open('task.csv', 'r') as f:
    lines = f.readlines()
s = int(lines[1])    
tens = np.load('tensor.npy')
kernel = np.load('kernel.npy')
bias = np.load('bias.npy')
res = conv(tens, kernel, bias, s)
np.save('seminar03_conv.npy', res, allow_pickle=False)