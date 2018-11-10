from keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from DataGeneration import DataGenerator
from skimage.transform import resize
from scipy.ndimage import grey_dilation

from keras.applications.vgg19 import VGG19

model_types = {'VGG19': VGG19}

##############################
#           set path
##############################
dataPath = '../../data'
if not os.path.exists(dataPath):
    os.makedirs(dataPath)

##############################
#           set params
##############################
testType, testNo = 'Test', '01'
repIs = 100
cMatchD, thr_xy = 20, 0.
model_reload, res_recalc = True, False

expTypes, expNo = {'A', 'B'}, {'A': '01', 'B': '01'}
sampleN_test = int(1000)
neuronN = [20]
imageSize, windowSize = [328, 328, 3], [128, 128]
windowStride = [4, 4]
outputSize = {'A': [1], 'B': [50]}
plt.ion()

test_gen = DataGenerator(psf_dir='../../data/exp_psf/psf 512_512.tif', expType=testType, sampleN=sampleN_test, pos_type='rand',
                         imageSize=imageSize, show_samples=False)
test_gen.imdb.fixXY = 0.
test_gen.imdb.neuronN = neuronN
test_gen.imdb.xyzis = test_gen.imdb.add_data()
test_gen.dataGeneratorOpts.areaL = [imageSize[0]-windowSize[0], imageSize[1]-windowSize[1]]

imgs = np.zeros([repIs, imageSize[0], imageSize[1], imageSize[2]])
for i in range(repIs):
    start_num, end_num = i, i+1
    img, _, _ = test_gen.generate_batch(start_num=start_num, end_num=end_num)
    imgs[start_num:end_num, :, :, :] = img

##############################
#           set GPU
##############################
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

for model_type, model_name in model_types.items():
    if model_reload:
        model = dict()
        for expType in expTypes:
            ##############################
            #           load model
            ##############################
            cacheDir = dataPath + '/Bead_' + expType + expNo[expType]
            modelFile = cacheDir + '/model' + '_' + model_type + '.h5'
            model[expType] = load_model(modelFile)

    res = {'tp': 0, 'fp': 0, 'fn': 0, 'pr': 0, 'rc': 0, 'jac': 0, 'mae': 0, 'rmse': 0}
    for repI in range(0, repIs):
        # print('Testing: %d of %d ...' % (repI, repIs))

        img = imgs[repI, :, :, :]
        pred = dict()
        preName = '_' + model_type + '_' + ('%03d' % repI)
        for expType in expTypes:
            resDir = dataPath + '/Bead_' + testType + testNo
            if not os.path.exists(resDir):
                os.makedirs(resDir)

            if expNo[expType] == '02':
                resFile = resDir + '/res' + preName + '_' + expType + '.npy'
            else:
                resFile = resDir + '/res' + preName + '_' + expType + expNo[expType] + '.npy'

            if os.path.exists(resFile) & (not res_recalc):
                pred[expType] = np.load(resFile)
            else:
                rcN = np.floor((np.array(img.shape[0:2]) - np.array(windowSize)) / np.array(windowStride)) + 1
                pred[expType] = np.zeros([int(np.prod(rcN)), outputSize[expType][0]])

                batchSize = 8
                batch = np.zeros([batchSize, windowSize[0], windowSize[1], 3])
                batchIdx = np.zeros([batchSize, ]).astype('int')
                rcIdx = np.array(range(0, int(np.prod(rcN))))
                test_gen.imdb.rng.shuffle(rcIdx)
                for rcI in range(0, int(np.prod(rcN))):
                    ri = rcIdx[rcI] // rcN[1]
                    ci = rcIdx[rcI] % rcN[1]
                    patch = img[int(ri*windowStride[0]):int(ri*windowStride[0]+windowSize[0]),
                                int(ci*windowStride[1]):int(ci*windowStride[1]+windowSize[1])]

                    patch = test_gen.my_cut(patch, 0.001, 0.999)
                    patch = (patch - np.min(patch)) / (np.max(patch) - np.min(patch))

                    batchI = rcI % batchSize
                    batch[batchI, :, :, :] = patch
                    batchIdx[batchI] = rcIdx[rcI]

                    if (batchI == batchSize-1) | (rcI == np.prod(rcN)-1):
                        im = batch[0:batchI+1, :, :, :]
                        pred_batch = model[expType].predict_on_batch(im)
                        pred[expType][batchIdx, :] = pred_batch

                pred[expType] = np.reshape(pred[expType], [int(rcN[0]), int(rcN[1]), -1])
                np.save(resFile, pred[expType])

        pred_xy = np.squeeze(resize(pred['A'], [img.shape[0]-windowSize[0], img.shape[1]-windowSize[1]]))
        pred_z = resize(pred['B'], [img.shape[0]-windowSize[0], img.shape[1]-windowSize[1], pred['B'].shape[2]])

        local_max = (grey_dilation(pred_xy, (3, 3)) == pred_xy)
        if thr_xy >= 1:
            xx, yy = np.where(local_max)
            values = []
            for ii in range(xx.shape[0]):
                values.append(pred_xy[xx[ii], yy[ii]])
            v_sorted = np.argsort(np.array(values))
            v_sorted = v_sorted[-1:-np.min([values.__len__(), thr_xy])-1:-1]
            xx, yy = xx[v_sorted], yy[v_sorted]
            maxZ = np.argmax(np.reshape(pred_z, [pred_z.shape[0]*pred_z.shape[1], -1]), 1)
            zz = maxZ[xx*local_max.shape[1]+yy]
        else:
            local_max = local_max & (pred_xy >= thr_xy)
            xx, yy = np.where(local_max)
            maxZ = np.argmax(np.reshape(pred_z, [pred_z.shape[0]*pred_z.shape[1], -1]), 1)
            zz = maxZ[np.where(np.reshape(local_max, [local_max.shape[0]*local_max.shape[1], -1]))[0]]
        centers = np.array([xx, yy, zz]).transpose()

        xyzs = test_gen.imdb.xyzis[repI][:, 0:3]
        areaL = np.array([test_gen.dataGeneratorOpts.areaL[0], test_gen.dataGeneratorOpts.areaL[1], outputSize['B'][0]*2])
        gts = np.abs(np.multiply(xyzs, areaL/2) + np.array([areaL[0]/2, areaL[1]/2, 0]))

        if repI == 1:
            img_cut = img[int(windowSize[0]/2):-int(windowSize[0]/2), int(windowSize[0]/2):-int(windowSize[0]/2), :]
            fig = plt.figure()
            plt.imshow(img_cut)
            plt.plot(centers[:, 1], centers[:, 0], 'gs', c=[0, 1, 0], markerfacecolor='none', markersize=8, markeredgewidth=2)
            plt.plot(gts[:, 1], gts[:, 0], 'r.', markersize=10)
            plt.draw()
            figFile = resDir + '/fig' + '_' + ('%03d' % repI) + '_' + model_type + '.png'
            fig.savefig(figFile)
            plt.close()

        dists = centers.reshape([centers.shape[0], 1, -1]) - gts.reshape([1, gts.shape[0], -1])
        dists = np.sum(dists**2, 2)**.5
        pairs = []
        for i in range(0, min(dists.shape)):
            if min(dists.reshape([-1, 1])) > cMatchD:
                break
            idx = np.argmin(dists.reshape([-1, 1]))
            xi = idx // dists.shape[1]
            yi = idx % dists.shape[1]

            pairs.append([xi, yi])
            dists[xi, :] = np.Inf
            dists[:, yi] = np.Inf

        pairs = np.array(pairs)
        for i in range(0, pairs.shape[0]):
            locs = np.array([centers[pairs[i, 0], :], gts[pairs[i, 1], :]])
            ae = np.abs(locs[0, :] - locs[1, :])

            res['mae'] = (res['mae']*res['tp']+ae)/(res['tp']+1)
            res['rmse'] = ((res['rmse']**2*res['tp']+ae**2)/(res['tp']+1))**.5
            res['tp'] = res['tp'] + 1
        res['fp'] = res['fp'] + centers.shape[0] - pairs.shape[0]
        res['fn'] = res['fn'] + gts.shape[0] - pairs.shape[0]

        res['pr'] = res['tp'] / (res['tp'] + res['fp'])
        res['rc'] = res['tp'] / (res['tp'] + res['fn'])
        res['jac'] = res['tp'] / (res['tp'] + res['fp'] + res['fn'])


    print('%s \t- jac: %.1f, pr: %.1f, rc: %.1f, rmse: %.2f, mae: %.2f' % (model_type, res['jac']*100, res['pr']*100,
          res['rc']*100, np.mean(res['rmse']**2)**.5, np.mean(res['mae'])))

sess.close()

# plt.pause(1e3)
