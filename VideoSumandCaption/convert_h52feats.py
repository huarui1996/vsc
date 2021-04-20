import h5py
import numpy as np

f = h5py.File("G:/pytorch-video-feature-extractor/features/msvd.h5")

keys = f.keys()
sampler = 40

max_len = 0
for i in keys:
    samples = np.round(np.linspace(0, f[i][()].shape[0] - 1, sampler)).astype(np.int32)
    data = f[i][()][samples]
    np.save('feats/' + str(i) + '.npy', data)
