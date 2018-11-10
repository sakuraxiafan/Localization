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
dataPath = './data'
if not os.path.exists(dataPath):
    os.makedirs(dataPath)

##############################
#           set params
##############################
testType, testNo = 'Test', '01'
repIs = 100
cMatchD, thr_xy = 200, 0.
model_reload, res_recalc = True, False

expTypes, expNo = {'A', 'B'}, {'A': '01', 'B': '01'}
sampleN_test = int(1000)
neuronN = [20]
imageSize, windowSize = [64, 64, 3], [13, 13]
windowStride = [1, 1]
outputSize = [1, 100, 100, 100]
plt.ion()

test_gen = DataGenerator(sampleN=sampleN_test, expType=testType, imageSize=imageSize, outputSize=outputSize, show_samples=False)

imgs = np.zeros([repIs, imageSize[0], imageSize[1], imageSize[2]])
for i in range(repIs):
    start_num, end_num = i, i+1
    img, _, _, _, _  = test_gen.generate_batch(start_num=start_num, end_num=end_num)
    imgs[start_num:end_num, :, :, :] = img

##############################
#           set GPU
##############################
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

for model_type, model_name in model_types.items():
    if model_reload:
        model = dict()
        for expType in expTypes:
            ##############################
            #           load model
            ##############################
            cacheDir = dataPath + '/SR_' + expType + expNo[expType]
            modelFile = cacheDir + '/model' + '_' + model_type + '.h5'
            model[expType] = load_model(modelFile)

    res = {'tp': 0, 'fp': 0, 'fn': 0, 'pr': 0, 'rc': 0, 'jac': 0, 'mae': 0, 'rmse': 0}
    for repI in range(0, repIs):
        # print('Testing: %d of %d ...' % (repI, repIs))

        img = imgs[repI, :, :, :]
        rcN = np.floor((np.array(img.shape[0:2]) - np.array(windowSize)) / np.array(windowStride)) + 1
        pred_xy = np.zeros([int(np.prod(rcN)), outputSize[0]])
        pred_x = np.zeros([int(np.prod(rcN)), outputSize[1]])
        pred_y = np.zeros([int(np.prod(rcN)), outputSize[2]])
        pred_z = np.zeros([int(np.prod(rcN)), outputSize[3]])
        preName = '_' + model_type + '_' + ('%03d' % repI)
        for expType in expTypes:
            resDir = dataPath + '/test_' + testType + testNo
            if not os.path.exists(resDir):
                os.makedirs(resDir)

            resFile = resDir + '/res' + preName + '_' + expType + expNo[expType] + '.npz'
            if os.path.exists(resFile) & (not res_recalc):
                pred = np.load(resFile)
                if expType == 'A':
                    data = np.load(resFile)
                    pred_xy = data['pred_xy']
                elif expType == 'B':
                    pred_x, pred_y, pred_z = np.load(resFile)
                    data = np.load(resFile)
                    pred_x, pred_y, pred_z = data['pred_x'], data['pred_y'], data['pred_z']
            else:
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

                    # patch = (patch - np.min(patch)) / (np.max(patch) - np.min(patch))

                    batchI = rcI % batchSize
                    batch[batchI, :, :, :] = patch
                    batchIdx[batchI] = rcIdx[rcI]

                    if (batchI == batchSize-1) | (rcI == np.prod(rcN)-1):
                        im = batch[0:batchI+1, :, :, :]
                        pred_batch = model[expType].predict_on_batch(im)
                        if expType == 'A':
                            pred_xy[batchIdx, :] = pred_batch
                        elif expType == 'B':
                            pred_x[batchIdx, :] = pred_batch[0]
                            pred_y[batchIdx, :] = pred_batch[1]
                            pred_z[batchIdx, :] = pred_batch[2]

                if expType == 'A':
                    pred_xy = np.squeeze(resize(np.reshape(pred_xy, [int(rcN[0]), int(rcN[1]), -1]), [img.shape[0]-windowSize[0], img.shape[1]-windowSize[1]]))
                elif expType == 'B':
                    pred_x = resize(np.reshape(pred_x, [int(rcN[0]), int(rcN[1]), -1]), [img.shape[0]-windowSize[0], img.shape[1]-windowSize[1], pred_x.shape[-1]])
                    pred_y = resize(np.reshape(pred_y, [int(rcN[0]), int(rcN[1]), -1]), [img.shape[0]-windowSize[0], img.shape[1]-windowSize[1], pred_x.shape[-1]])
                    pred_z = resize(np.reshape(pred_z, [int(rcN[0]), int(rcN[1]), -1]), [img.shape[0]-windowSize[0], img.shape[1]-windowSize[1], pred_x.shape[-1]])

                if expType == 'A':
                    np.savez(resFile, pred_xy=pred_xy)
                elif expType == 'B':
                    np.savez(resFile, pred_x=pred_x, pred_y=pred_y, pred_z=pred_z)

        local_max = (grey_dilation(pred_xy, (3, 3)) == pred_xy)
        if thr_xy >= 1:
            xx, yy = np.where(local_max)
            values = []
            for ii in range(xx.shape[0]):
                values.append(pred_xy[xx[ii], yy[ii]])
            v_sorted = np.argsort(np.array(values))
            v_sorted = v_sorted[-1:-np.min([values.__len__(), thr_xy])-1:-1]
            xx, yy = xx[v_sorted], yy[v_sorted]
            maxX = np.argmax(np.reshape(pred_x, [pred_x.shape[0]*pred_x.shape[1], -1]), 1)
            maxY = np.argmax(np.reshape(pred_y, [pred_y.shape[0]*pred_y.shape[1], -1]), 1)
            maxZ = np.argmax(np.reshape(pred_z, [pred_z.shape[0]*pred_z.shape[1], -1]), 1)
            dx = maxX[xx*local_max.shape[1]+yy]
            dy = maxY[xx*local_max.shape[1]+yy]
            zz = maxZ[xx*local_max.shape[1]+yy]
        else:
            local_max = local_max & (pred_xy >= thr_xy)
            xx, yy = np.where(local_max)
            maxX = np.argmax(np.reshape(pred_x, [pred_x.shape[0]*pred_x.shape[1], -1]), 1)
            maxY = np.argmax(np.reshape(pred_y, [pred_y.shape[0]*pred_y.shape[1], -1]), 1)
            maxZ = np.argmax(np.reshape(pred_z, [pred_z.shape[0]*pred_z.shape[1], -1]), 1)
            dx = maxX[np.where(np.reshape(local_max, [local_max.shape[0]*local_max.shape[1], -1]))[0]]
            dy = maxY[np.where(np.reshape(local_max, [local_max.shape[0]*local_max.shape[1], -1]))[0]]
            zz = maxZ[np.where(np.reshape(local_max, [local_max.shape[0]*local_max.shape[1], -1]))[0]]

        outR = test_gen.imdb.xyzRange
        centers = np.array([(xx+dx/outputSize[1]*(outR[0, 1]-outR[0, 0])+outR[0, 0]+windowSize[0]/2)*test_gen.imdb.xyzScale[0],
                            (yy+dy/outputSize[1]*(outR[1, 1]-outR[1, 0])+outR[1, 0]+windowSize[1]/2)*test_gen.imdb.xyzScale[1],
                            (zz/outputSize[3]*(outR[2, 1]-outR[2, 0])+outR[2, 0])*test_gen.imdb.xyzScale[2]]).transpose()

        gts = np.array([(test_gen.imdb.gt_bg['x'][repI,:]+test_gen.imdb.dx)*test_gen.imdb.xyzScale[0],
                        (test_gen.imdb.gt_bg['y'][repI,:]+test_gen.imdb.dx)*test_gen.imdb.xyzScale[1],
                        (test_gen.imdb.gt_bg['z'][repI,:])*test_gen.imdb.xyzScale[2]]).transpose()

        if repI == 1:
            img_cut = img[int(windowSize[0]/2):-int(windowSize[0]/2), int(windowSize[0]/2):-int(windowSize[0]/2), :]
            fig = plt.figure()
            plt.imshow(img/np.max(img))
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
