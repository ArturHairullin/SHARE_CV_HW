from PIL import Image
import numpy as np
img = Image.open('image.png')
imar = np.array(img)
imar = imar.astype(np.float32)
for i in range(0,32):
    for j in range(0,32):
        for k in range(0,3):
            imar[i][j][k]=imar[i][j][k]/255.
ar = imar.mean(axis=2)
with open('task.csv', 'r') as f:
    lines = f.readlines()
r, c, l = map(int, lines[1].split(','))
patch = ar[r:r+l, c:c+l]
np.save('seminar01_crop.npy', patch, allow_pickle=False)