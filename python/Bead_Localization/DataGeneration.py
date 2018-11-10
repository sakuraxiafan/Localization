import numpy as np
import tifffile as tiff
from parameters import Imdb
import h5py
from matplotlib import pyplot as plt
import time
import os


class DataGeneratorOpts(object):
    def __init__(self, imageSize=[128,128,3], outputSize=[1]):
        self.imageSize = imageSize
        self.outputSize = outputSize
        self.max_d = np.array([4, 4])
        self.r = 0
        self.normFlag = 1
        self.areaL = [256, 256]
        self.nois = 0.1                # relative noise factor to determine the noise level.   0.7 roughly corresponding to 3dB


class DataGenerator(object):
    def __init__(self, psf_dir='../../data/exp_psf/psf.tif', sampleN=int(1e3), pos_type='randn',
                 expType='A', imageSize=[128,128,3], outputSize=[1], show_samples=False):
        self.imdb = Imdb(sampleN, expType, pos_type)
        self.dataGeneratorOpts = DataGeneratorOpts(imageSize, outputSize)
        self.psf_kernel = self.load_psf(psf_dir)
        self.expType = expType
        self.show_samples = show_samples

    def load_psf(self, psf_dir):
        psf_kernel = tiff.imread(psf_dir)
        psf_kernel = np.transpose(psf_kernel, [1, 2, 0])
        psfMaxZ = 50
        psfC = 26 - 1  # python index start from 0
        dz = 1
        psf_slice = np.concatenate([np.arange(start=psfC + psfMaxZ * dz, step=-dz, stop=psfC, dtype=np.int),
                                    np.arange(start=psfC, step=dz, stop=psfC + psfMaxZ * dz + 1, dtype=np.int)], axis=0)
        psf_kernel = psf_kernel[:, :, psf_slice]
        return psf_kernel

    def generator(self, batch_size=8):
        data_num = self.imdb.sampleN
        while 1:
            for i in range(int(data_num/batch_size)):
                start_num, end_num = i*batch_size, (i+1)*batch_size
                imgs, labels_xy, labels_z = self.generate_batch(start_num=start_num, end_num=end_num)
                if self.expType == 'A':
                    yield imgs, labels_xy
                elif self.expType == 'B':
                    yield imgs, labels_z

    def generate_batch(self, save_dir='', start_num=0, end_num=20000, pid=0):
        # areas = np.array([self.dataGeneratorOpts.imageSize[0] * 2, self.dataGeneratorOpts.imageSize[1] * 2, np.size(self.psf_kernel, 2)])
        areas = np.array([self.psf_kernel.shape[0], self.psf_kernel.shape[1], np.size(self.psf_kernel, 2)])
        prob_temp = self.imdb.rng.rand(1)<0.75 if self.imdb.pos_type == 'randn' else 0
        area = np.array([self.dataGeneratorOpts.areaL[0] / (2 ** (int(prob_temp) * 2)), self.dataGeneratorOpts.areaL[1] / (2 ** (int(prob_temp) * 2)), np.size(self.psf_kernel, 2)])
        rect = np.array([self.dataGeneratorOpts.imageSize[0], self.dataGeneratorOpts.imageSize[1]])

        batch_size = end_num-start_num
        imageSize = self.dataGeneratorOpts.imageSize
        outputSize = self.dataGeneratorOpts.outputSize
        imgs = np.zeros([batch_size, imageSize[0], imageSize[1], imageSize[2]])
        labels_z = np.zeros([batch_size, outputSize[0]])
        labels_xy = np.zeros([batch_size, outputSize[0]])
        start_time_pro = time.time()
        for ind in range(start_num, end_num):
            xyzi = self.imdb.xyzis[ind]
            img, x = self.generate_image_from_psf(xyzi=xyzi, h=self.psf_kernel, normFlag=self.dataGeneratorOpts.normFlag,
                                                  r=self.dataGeneratorOpts.r, areas=areas, area=area, rect=rect)
            label_xy, label_z = self.mark_label_xyz(x)
            im = np.tile(img[:, :, np.newaxis], (1, 1, 3))

            if not save_dir == '':
                save_data = save_dir + 'data_file/'
                if not os.path.exists(save_data): os.makedirs(save_data)
                self.save_h5py(im, label_xy, label_z, save_data + 'data_%06d.h5' % ind)

                save_image = save_dir + 'image_file/'
                if not os.path.exists(save_image): os.makedirs(save_image)
                self.draw_image(img, save_image + 'data_%06d.png' % ind)

                if (ind - start_num + 1) % 10 == 0:
                    end_time_pro = time.time()
                    print('Process %d has finished %d of %d datas for %.2f s' %(pid, ind - start_num + 1, end_num - start_num, end_time_pro - start_time_pro))
                    start_time_pro = time.time()

            else:
                imgs[ind - start_num, :, :, :] = im
                labels_xy[ind - start_num, :] = label_xy
                labels_z[ind - start_num, :] = np.squeeze(label_z)

        if self.show_samples:
            ms, ns = 2, 4
            plt.figure(99)
            for m in range(0, ms):
                for n in range(0, ns):
                    if m*ns+n+1 > batch_size:
                        continue
                    plt.subplot(ms, ns, m*ns+n+1)
                    plt.imshow(imgs[m*ns+n, :, :, 1])
                    plt.axis('off')
                    plt.title('%d, %s' % (labels_xy[m*ns+n, 0], str(np.where(labels_z[m*ns+n, :] != 0)[0].tolist())))
            plt.pause(10)

        if save_dir == '':
            return imgs, labels_xy, labels_z


    def save_h5py(self, img, labels_xy, labels_z, fileName):
        f = h5py.File(fileName, 'w')
        f['img'] = img
        f['labels_xy'] = labels_xy
        f['labels_z'] = labels_z
        f.close()

    def draw_image(self, img, imgName):
        img_temp = img / np.max(img)
        im = plt.imshow(img_temp, cmap='gray')
        plt.axis('off')
        plt.savefig(imgName)
        plt.clf()
        plt.close()

    def mark_label_xyz(self, x):
        #label_z = -np.ones(self.dataGeneratorOpts.outputSize)
        label_z = -np.ones([1, 1, self.dataGeneratorOpts.outputSize[0]]) * 0.
        d = self.dataGeneratorOpts.max_d
        xs = x[np.int(np.floor(np.size(x, 0) / 2) - d[0]): np.int(np.floor(np.size(x, 0) / 2) + d[0]),
                np.int(np.floor(np.size(x, 1) / 2) - d[1]): np.int(np.floor(np.size(x, 1) / 2) + d[1]), :]
        l = np.array(np.where(np.sum(np.sum(np.abs(xs), 1), 0) > 1e-7))
        j_begin = 0
        for j in range(1, np.size(l) + 1):
            if np.size(np.shape(l)) > 1:
                l = np.squeeze(l, axis=0)
            if j < np.size(l):
                if l[j] == l[j - 1] + 1:
                    continue
            l_vs = x[:, :, int(np.round((l[j_begin] + l[j - 1]) / 2))]
            l_xy = np.where(l_vs > 1e-7)
            l_x = l_xy[0][0]
            l_y = l_xy[1][0]
            l_r = np.min(((l_x - np.round(np.size(x, 0) / 2)) ** 2 + (l_y - np.round(np.size(x, 1) / 2)) ** 2) ** 0.5)  #???
            l_v = np.maximum(-1, np.minimum(1, 1 - l_r / np.sum(d ** 2) ** 0.5 * 2))
            l_scale = np.array([l[j_begin], l[j - 1]]) * 2 - np.size(x, 2) - 1
            l_cur = np.sort(np.floor(np.abs(l_scale) / np.size(x, 2) * self.dataGeneratorOpts.outputSize[0]))
            l_cur = np.array(l_cur, dtype=np.int)
            if np.prod(l_scale) < 0:
                l_cur[0] = 0
            label_z[:, :, l_cur[0]: l_cur[1] + 1] = np.maximum(label_z[:, :, l_cur[0]: l_cur[1] + 1], l_v)
            j_begin = j
        label_temp = np.squeeze(label_z)
        label_z[label_z!=0] = 1
        # label_z = np.argmax(label_z) + 1

        label_xy = np.any(label_temp!=0)
        if label_xy == 0:
            label_xy = -1
        return label_xy, label_z

    def generate_image_from_psf(self, xyzi, h, normFlag, r, areas, area, rect):
        x = np.zeros([areas[0], areas[1], areas[2]])
        xyzi_T = np.array([])
        for i in range(0, 3):
            temp = np.minimum(np.maximum(r + 1, np.round((area[i] - r) * xyzi[:, i] / 2 + np.size(x, i) / 2)), np.size(x, i) - r)
            temp = temp - 1       # matlab index convert to python index
            temp = np.reshape(temp, [np.size(temp, 0), 1])
            if np.size(xyzi_T) == 0:
                xyzi_T = temp
            else:
                xyzi_T = np.concatenate([xyzi_T, temp], axis=1)
        temp = np.reshape(xyzi[:, 3], [np.size(xyzi[:, 3], 0), 1])
        xyzi_T = np.concatenate([xyzi_T, temp], axis=1)

        for i in range(np.size(xyzi_T, 0)):
            x[np.int(xyzi_T[i, 0] + np.arange(-r, r + 1)), np.int(xyzi_T[i, 1] + np.arange(-r, r + 1)), np.int(xyzi_T[i, 2] + np.arange(-r, r + 1))] = xyzi_T[i, 3] / ((2 * r + 1) ** 3)

        I = np.where(np.abs(np.sum(np.sum(x, 1), 0)) >= 1e-10)
        I = np.squeeze(np.array(I))
        if np.size(I) == 1:
            I = np.reshape(np.array(I), [1])
        ys = np.zeros([np.size(x, 0), np.size(x, 1), np.size(I)])
        if np.size(I) > 0:
            for i in range(0, np.size(I)):
                # ys[:, :, i] = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.fft.fftshift(np.fft.fft2(x[:, :, I[i]]))) * np.fft.fftshift(np.fft.fft2(h[:, :, I[i]])))))
                [xx, yy] = np.where(x[:, :, I[i]] != 0)
                sz = [int(x.shape[0]/2), int(x.shape[1]/2)]
                for j in range(0, len(xx)):
                    ys[max(0,xx[j]-sz[0]):min(sz[0]*2-1,xx[j]+sz[0]-1), max(0,yy[j]-sz[1]):min(sz[1]*2-1,yy[j]+sz[1]-1), i] += \
                        h[max(0,-xx[j]+sz[0]):min(sz[0]*2-1,-xx[j]+sz[0]*3-1), max(0,-yy[j]+sz[1]):min(sz[1]*2-1,-yy[j]+sz[1]*3-1), I[i]] * x[xx[j], yy[j], I[i]]
                """
                ax = plt.gca()
                im = plt.imshow(ys[:, :, i])
                plt.axis('off')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
                plt.clf()
                """
        y_mid = np.sum(ys, axis=2)

        rate = [0.001, 0.999]
        if normFlag == -1:
            img = np.real(y_mid)
            img = self.my_cut(img, rate[0], rate[1])
        elif normFlag == 0:
            img = np.real(np.log(y_mid))
            img = self.my_thr(img, -20, 0.1)
            img = self.my_cut(img, rate[0], rate[1])
            img = np.maximum(0, (img+20)/20)
        elif normFlag == 1:
            img = np.real(y_mid)
            img = self.my_cut(img, rate[0], rate[1])
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
        else:
            img = np.real(np.log(y_mid))
            img = self.my_cut(img, rate[0], rate[1])
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = self.my_rect(img, rect)
        """
        if self.dataGeneratorOpts.nois:
            noise = self.add_noise(img=img, rnl=self.dataGeneratorOpts.nois)
            img = img + noise
            snr = 10 * np.log10(np.sum((img - noise) ** 2) / np.sum(noise ** 2))
        """
        return img, x

    def my_cut(self, x, rate1, rate2):
        y = x
        x_to_sort = np.reshape(x, [-1])
        v = np.sort(x_to_sort)
        m = np.array([v[np.int(np.maximum(0, np.round(rate1 * np.size(v))) - 1)], v[np.int(np.round(rate2 * np.size(v)) - 1)]])
        y[np.where(y<m[0])] = m[0]
        y[np.where(y>m[1])] = m[1]
        return y

    def my_thr(self, x, thr, rate):
        y = x
        tmp = x[np.where(x>=thr)]
        sorted_tmp = np.sort(tmp)
        value = sorted_tmp[np.floor(rate * np.size(sorted_tmp))]
        y[np.where(x<=value)] = value
        return y

    def my_rect(self, img, rect):
        sizes = np.array([np.size(img, 0), np.size(img, 1)])
        region = np.maximum(1, np.floor(sizes - rect) / 2) - 1                       # matlab and python index is different
        region = np.array(np.concatenate([region, region + rect]), dtype=np.int)
        img = img[region[0]: region[2], region[1]: region[3]]
        return img

    def add_noise(self, img, rnl):
        e = np.random.random([np.size(img, 0), np.size(img, 1)])
        e = e / np.linalg.norm(e)
        e = rnl * np.linalg.norm(img) * e
        return e

def generation_multiprocess(demo, save_dir, start_num, end_num, pid):
    return demo.generate_batch(save_dir, start_num, end_num, pid)


if __name__ == '__main__':

    dataGenerator = DataGenerator('../../data/exp_psf/psf.tif', expType='A')
    dataGenerator.generate_batch('../../data/Bead_training_netA_data/', start_num=0, end_num=10)

    dataGenerator = DataGenerator('../../data/exp_psf/psf.tif', expType='B')
    dataGenerator.generate_batch('../../data/Bead_training_netB_data/', start_num=0, end_num=10)

    print('Data Generation Finished!')
