import numpy as np
import scipy.io


class Imdb(object):
    def __init__(self, sampleN=int(1e3), expType='A', psf_dir='../../data/exp_psf/bead_astig_3dcal.mat', imageSize=13):
        self.sampleN = np.int(sampleN)
        self.expType = expType
        self.psf = scipy.io.loadmat(psf_dir)
        self.rng = np.random.RandomState(self.sampleN)

        self.RoiPixelsize = imageSize
        self.dz = np.int(self.psf['cspline']['dz'])
        self.z0 = np.int(self.psf['cspline']['z0'])
        self.dx = self.RoiPixelsize / 2
        self.xyzRange = np.array([[-1, 1], [-1, 1], [-500, 500]])
        self.xyzScale = [1, 1, 1/132]
        self.fixXY = 1.
        self.cPixel = 2

        self.Intensity = 2000
        self.background = 10

        self.bgRange = 1
        self.density = np.array([2])

        if self.expType == 'A':
            self.fixXY = .5
            self.bgRange = 2
            self.density = np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5])
        elif self.expType == 'B':
            self.fixXY = 1.
            self.bgRange = 1.5
            self.density = np.array([2])
        elif self.expType == 'T':
            self.fixXY = .0
            self.bgRange = (64-12)/64
            self.density = np.array([20])

        self.gt_bg, self.cor_bg = self.add_data()

    def add_data(self):
        densities = self.density[self.rng.randint(0, np.size(self.density), [self.sampleN, 1])]
        gt_bg = {'x': np.array([]), 'y': np.array([]), 'z': np.array([])}
        cor_bg = np.array([])
        for i in range(max(self.density)):
            x_tmp = (self.rng.rand(self.sampleN, 1) * (self.xyzRange[0, 1]-self.xyzRange[0, 0]) + self.xyzRange[0, 0]) \
                    * self.bgRange * self.RoiPixelsize / 2
            y_tmp = (self.rng.rand(self.sampleN, 1) * (self.xyzRange[1, 1]-self.xyzRange[1, 0]) + self.xyzRange[1, 0]) \
                    * self.bgRange * self.RoiPixelsize / 2
            z_tmp = self.rng.rand(self.sampleN, 1) * (self.xyzRange[2, 1]-self.xyzRange[2, 0]) + self.xyzRange[2, 0]

            if i == 0:
                fixMask = self.rng.rand(self.sampleN, 1) < self.fixXY
                x_tmp[fixMask] = x_tmp[fixMask] / self.RoiPixelsize
                y_tmp[fixMask] = y_tmp[fixMask] / self.RoiPixelsize

            zeroMask = densities <= i
            x_tmp[zeroMask] = 0
            y_tmp[zeroMask] = 0
            z_tmp[zeroMask] = 0

            if i == 0:
                gt_bg['x'] = x_tmp
                gt_bg['y'] = y_tmp
                gt_bg['z'] = z_tmp
                cor_bg = np.hstack([x_tmp + self.dx, y_tmp + self.dx, z_tmp / self.dz + self.z0])
            else:
                gt_bg['x'] = np.concatenate((gt_bg['x'], x_tmp), 1)
                gt_bg['y'] = np.concatenate((gt_bg['y'], y_tmp), 1)
                gt_bg['z'] = np.concatenate((gt_bg['z'], z_tmp), 1)
                cor_bg = np.concatenate((cor_bg, np.hstack([x_tmp + self.dx, y_tmp + self.dx, z_tmp / self.dz + self.z0])), 1)

        return gt_bg, cor_bg

if __name__ == '__main__':
    imdb = Imdb(1e3, '../../data/exp_psf/bead_astig_3dcal.mat')
    print('Data Generation Finished!')
