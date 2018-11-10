import numpy as np
from parameters import Imdb
import h5py
from matplotlib import pyplot as plt
import os


class DataGenerator(object):
    def __init__(self, psf_dir='../../data/exp_psf/bead_astig_3dcal.mat', sampleN=int(1e3), expType='A',
                 imageSize=[13, 13, 3], outputSize=[1, 100, 100, 100], show_samples=False):
        self.imdb = Imdb(sampleN, expType, psf_dir, imageSize[0])
        self.expType = expType
        self.imageSize = imageSize
        self.outputSize = outputSize
        self.xyzScale = [1./132, 1./132, 1.]
        self.show_samples = show_samples

    def generator(self, batch_size=8):
        data_num = self.imdb.sampleN
        while 1:
            for i in range(int(data_num/batch_size)):
                start_num, end_num = i*batch_size, (i+1)*batch_size
                imgs, labels_xy, labels_x, labels_y, labels_z = self.generate_batch(start_num=start_num, end_num=end_num)
                if self.expType == 'A':
                    yield imgs, labels_xy
                elif self.expType == 'B':
                    yield ({'input_1': imgs}, {'out_x': labels_x, 'out_y': labels_y, 'out_z': labels_z})

    def generate_batch(self, save_dir='', start_num=0, end_num=20000, pid=0):
        batch_size = end_num-start_num
        imageSize = self.imageSize
        outputSize = self.outputSize
        imgs = np.zeros([batch_size, imageSize[0], imageSize[1], imageSize[2]])
        labels_xy = np.zeros([batch_size, outputSize[0]])
        labels_x = np.zeros([batch_size, outputSize[1]])
        labels_y = np.zeros([batch_size, outputSize[2]])
        labels_z = np.zeros([batch_size, outputSize[3]])
        for ind in range(start_num, end_num):
            xyzi = self.imdb.cor_bg[ind, :]
            img, label_xy, label_x, label_y, label_z = self.generate_image_from_psf(ind)
            im = np.tile(img, (1, 1, 3))

            imgs[ind - start_num, :, :, :] = im
            labels_xy[ind - start_num, :] = label_xy
            labels_x[ind - start_num, :] = np.squeeze(label_x)
            labels_y[ind - start_num, :] = np.squeeze(label_y)
            labels_z[ind - start_num, :] = np.squeeze(label_z)

            if not save_dir == '':
                save_data = save_dir + 'data_file/'
                if not os.path.exists(save_data): os.makedirs(save_data)
                self.save_h5py(im, label_xy, label_x, label_y, label_z, save_data + 'data_%06d.h5' % ind)

                save_image = save_dir + 'image_file/'
                if not os.path.exists(save_image): os.makedirs(save_image)
                self.draw_image(im, save_image + 'data_%06d.png' % ind)

                if (ind - start_num + 1) % 10 == 0:
                    print('Process %d has finished %d of %d datas' %(pid, ind - start_num + 1, end_num - start_num))

        if self.show_samples:
            ms, ns = 2, 4
            plt.figure(99)
            for m in range(0, ms):
                for n in range(0, ns):
                    plt.subplot(ms, ns, m*ns+n+1)
                    plt.imshow(imgs[m*ns+n, :, :, :]/np.max(imgs[m*ns+n, :, :, :]), interpolation='none')
                    plt.axis('off')
                    plt.title('%d, %s, %s, %s' % (labels_xy[m*ns+n, 0],
                                                  str(np.where(labels_x[m * ns + n, :] != 0)[0].tolist()),
                                                  str(np.where(labels_y[m * ns + n, :] != 0)[0].tolist()),
                                                  str(np.where(labels_z[m * ns + n, :] != 0)[0].tolist())))
            plt.pause(10)

        if save_dir == '':
            return imgs, labels_xy, labels_x, labels_y, labels_z

    def generate_image_from_psf(self, idx, overlapFlag=False):
        cor = self.imdb.cor_bg
        coeff = self.imdb.psf['cspline']['coeff'][0,0]
        Npixels = self.imdb.RoiPixelsize
        outR = self.imdb.xyzRange

        Nfits = np.size(idx)
        spline_xsize = np.shape(coeff)[0]
        spline_ysize = np.shape(coeff)[1]
        spline_zsize = np.shape(coeff)[2]
        off = np.floor(((spline_xsize + 1) - Npixels) / 2)
        data = np.zeros([Npixels, Npixels, Nfits], 'single') + self.imdb.background

        label_xy = np.zeros([Nfits, self.outputSize[0]], 'single')
        label_x = np.zeros([Nfits, self.outputSize[1]], 'single')
        label_y = np.zeros([Nfits, self.outputSize[2]], 'single')
        label_z = np.zeros([Nfits, self.outputSize[3]], 'single')

        density = np.int(np.shape(cor)[1] / 3)
        masks = np.zeros([Npixels, Npixels, density, Nfits], 'bool')

        for ki in range(Nfits):
            if Nfits == 1:
                kk = idx
            else:
                kk = idx[ki]

            for di in range(density):
                xcenter = cor[kk, 0 + di * 3]
                ycenter = cor[kk, 1 + di * 3]
                zcenter = cor[kk, 2 + di * 3]

                # xc = -1 * (xcenter - Npixels / 2. + 0.5)
                # yc = -1 * (ycenter - Npixels / 2. + 0.5)
                xc = -1 * (xcenter - Npixels / 2.)
                yc = -1 * (ycenter - Npixels / 2.)
                zc = zcenter - np.floor(zcenter)

                xstart = np.floor(xc)
                xc = xc - xstart
                ystart = np.floor(yc)
                yc = yc - ystart
                zstart = np.floor(zcenter)

                if di >= 1 and ((cor[kk, 0] - xcenter) ** 2 + (cor[kk, 1] - ycenter) ** 2) ** .5 <= 3:
                    if ~overlapFlag:
                        continue

                if xcenter == Npixels / 2. and ycenter == Npixels / 2.:
                    continue

                delta_f = self.computeDelta3Dj_v2(np.single(xc), np.single(yc), np.single(zc))

                for ii in range(Npixels):
                    for jj in range(Npixels):
                        temp = self.fAt3Dj_v2(np.int(ii+xstart+off), np.int(jj+ystart+off), np.int(zstart),
                                              spline_xsize, spline_ysize, spline_zsize, delta_f, coeff)
                        model = temp * self.imdb.Intensity
                        data[ii, jj, ki] = data[ii, jj, ki] + model
                        if temp > 5e-3:
                            masks[ii, jj, di, ki] = 1

            xs, ys = self.imdb.gt_bg['x'][kk, :], self.imdb.gt_bg['y'][kk, :]
            zeroMask = (xs != 0) & (ys != 0)
            xs, ys = xs[zeroMask], ys[zeroMask]
            label_xy[ki, :] = np.int(any(xs**2+ys**2 <= self.imdb.cPixel**2)) * 2 - 1

            lab_x = np.int(round(max(0, min(1, (self.imdb.gt_bg['x'][kk, 0] - outR[0, 0]) / (outR[0, 1] - outR[0, 0])))
                                 * (self.outputSize[1] - 1)))
            lab_y = np.int(round(max(0, min(1, (self.imdb.gt_bg['y'][kk, 0] - outR[1, 0]) / (outR[1, 1] - outR[1, 0])))
                                 * (self.outputSize[2] - 1)))
            lab_z = np.int(round((self.imdb.gt_bg['z'][kk, 0] - outR[2, 0]) / (outR[2, 1] - outR[2, 0]) * (self.outputSize[3] - 1)))
            label_x[ki, lab_x], label_y[ki, lab_y], label_z[ki, lab_z] = 1, 1, 1

        img = self.imdb.rng.poisson(data)
        # img = img / np.max(img)

        return img, label_xy, label_x, label_y, label_z

    def computeDelta3Dj_v2(self, x_delta, y_delta, z_delta):
        delta_f = np.zeros([64,])
        cz = 1
        for i in range(4):
            cy = 1
            for j in range(4):
                cx = 1
                for k in range(4):
                    delta_f[i*16+j*4+k] = cx*cy*cz
                    cx = cx*x_delta
                cy = cy*y_delta
            cz = cz*z_delta
        return delta_f

    def fAt3Dj_v2(self, xc, yc, zc, xsize, ysize, zsize, delta_f, coeff):
        xc = max(xc, 0)
        xc = min(xc, xsize-1)
        yc = max(yc, 0)
        yc = min(yc, ysize-1)
        zc = max(zc, 0)
        zc = min(zc, zsize-1)
        temp = coeff[xc, yc, zc, :]
        pd = sum(np.multiply(delta_f, temp))
        return pd

    def save_h5py(self, img, label_xy, label_x, label_y, label_z, fileName):
        f = h5py.File(fileName, 'w')
        f['img'] = img
        f['label_xy'] = label_xy
        f['label_x'] = label_x
        f['label_y'] = label_y
        f['label_z'] = label_z
        f.close()

    def draw_image(self, img, imgName):
        img_temp = img / np.max(img)
        im = plt.imshow(img_temp, cmap='gray', interpolation='none')
        plt.axis('off')
        plt.savefig(imgName)
        plt.clf()
        plt.close()


if __name__ == '__main__':

    dataGenerator = DataGenerator('../../data/exp_psf/bead_astig_3dcal.mat', expType='A', sampleN=1e3, show_samples=False)
    dataGenerator.generate_batch('../../data/SR_training_netA_data/', start_num=0, end_num=10)

    dataGenerator = DataGenerator('../../data/exp_psf/bead_astig_3dcal.mat', expType='B', sampleN=1e3, show_samples=False)
    dataGenerator.generate_batch('../../data/SR_training_netB_data/', start_num=0, end_num=10)

    print('Data Generation Finished!')

