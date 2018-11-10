import numpy as np


class Imdb(object):
    def __init__(self, sampleN=int(1e3), expType='A', pos_type='randn'):
        self.sampleN = sampleN
        self.expType = expType
        self.pos_type = pos_type
        self.neuronN = [2, 5]
        self.fixXY = 0.2
        self.fixI = 0

        self.normFlag = 1
        self.r = 0
        self.max_d = [1, 1] * 4
        self.imageSize = [128, 128, 3]
        self.outputSize = [1, 1, 1]
        self.numAugments = 1
        self.areaL = np.min([self.imageSize[0], self.imageSize[1]]) * 2

        self.rng = np.random.RandomState(self.sampleN)

        if self.expType == 'A':
            self.fixXY = 0.2
            self.outputSize = [1, 1, 1]
        elif self.expType == 'B':
            self.fixXY = 1.
            self.outputSize = [1, 1, 50]
        elif self.expType == 'T':
            self.fixXY = 0.2
            self.outputSize = [1, 1, 1]

        self.xyzis = self.add_data()

    def add_data(self):
        xyzis = []
        rng = self.rng
        for ind in range(0, self.sampleN):
            n = rng.randint(low=self.neuronN[0], high=self.neuronN[-1] + 1)
            if self.pos_type == 'randn':
                tmp = rng.randn(n, 3)
                xyzi = np.concatenate([np.mod(tmp, np.sign(tmp)), rng.rand(n, 1) * 0.9 + 0.1], axis=1)
            elif self.pos_type == 'rand':
                tmp = (rng.rand(n, 3) * 2) - 1
                xyzi = np.concatenate([np.mod(tmp, np.sign(tmp)), rng.rand(n, 1) * 0. + 1.], axis=1)
            xyzis.append(xyzi)

        if self.fixXY:
            rands = rng.rand(self.sampleN, 1)
            for ind in range(0, self.sampleN):
                if rands[ind, 0] < self.fixXY:
                    xyzis[ind][range(0, np.ceil(self.fixXY).astype(int)), 0: 2] = 0
        return xyzis