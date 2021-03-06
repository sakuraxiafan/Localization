from keras.models import Model
from keras.layers import Dense, GlobalMaxPool2D, Dropout
from keras.layers import Input
from keras.optimizers import SGD
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from DataGeneration import DataGenerator

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
expType, expNo = 'A', '01'
net_continue = 0

nb_epoch, batch_size = 5, 8
sampleN = int(2e5)
imageSize = [128, 128, 3]
outputSize = [1]
show_samples = False
lr, momentum, decay = 0.001, 0.9, 5e-4
plt.ion()

##############################
#           set GPU
##############################
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3333)

for model_type, model_name in model_types.items():
    print('\n##############################\n Run for model: %s\n##############################' % model_type)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    ##############################
    #           get model
    ##############################
    cacheDir = dataPath + '/Bead_' + expType + expNo
    if not os.path.exists(cacheDir):
        os.makedirs(cacheDir)

    preName = '_' + model_type
    modelFile = cacheDir + '/model' + preName + '.h5'
    if os.path.exists(modelFile) & net_continue:
        ##############################
        #           load model
        ##############################
        model = load_model(modelFile)
    else:
        ##############################
        #           generate model
        ##############################
        base_model = model_name(input_tensor=Input(shape=imageSize), weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalMaxPool2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(outputSize[0])(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=SGD(lr=lr, momentum=momentum, decay=decay), loss='hinge', metrics=['accuracy'])

        var_sizes = []
        for i in range(np.shape(model.layers)[0]):
            var_sizes.append(model.layers[i].output_shape)

        sampleN_val = max(10, int(sampleN/1000))
        train_gen = DataGenerator(sampleN=sampleN, expType=expType, outputSize=outputSize, show_samples=show_samples)
        val_gen = DataGenerator(sampleN=sampleN_val, expType=expType, outputSize=outputSize)

        ##############################
        #           train model
        ##############################
        hist = model.fit_generator(generator=train_gen.generator(batch_size),
                                   validation_data=val_gen.generator(batch_size),
                                   steps_per_epoch=sampleN/batch_size,
                                   validation_steps=sampleN_val/batch_size,
                                   epochs=nb_epoch, callbacks=[ModelCheckpoint(modelFile)])

        ##############################
        #           save model
        ##############################
        model.save(modelFile)

        ##############################
        #           plot curve
        ##############################
        fig = plt.figure()
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.draw()

        figFile = cacheDir + '/fig' + preName + '_training' + '.png'
        fig.savefig(figFile)

        plt.pause(5)
        plt.close()

    sampleN_test = max(20, int(sampleN/100))
    test_gen = DataGenerator(sampleN=sampleN_test, expType=expType, outputSize=outputSize)

    data_num = test_gen.imdb.sampleN
    imgs = np.zeros([data_num, imageSize[0], imageSize[1], imageSize[2]])
    labels_z = np.zeros([data_num, outputSize[0]])
    labels_xy = np.zeros([data_num, outputSize[0]])
    preds = np.zeros([data_num, outputSize[0]])
    for i in range(int(data_num / batch_size)):
        start_num, end_num = i * batch_size, (i + 1) * batch_size
        if i == int(data_num / batch_size)-1:
            end_num = data_num
        img, label_xy, label_z = test_gen.generate_batch(start_num=start_num, end_num=end_num)
        pred = model.predict_on_batch(img)

        imgs[start_num:end_num, :, :, :] = img
        labels_xy[start_num:end_num, :] = label_xy
        labels_z[start_num:end_num, :] = label_z
        preds[start_num:end_num, :] = pred

    ms, ns = 3, 4
    fig = plt.figure()
    for mi in range(ms):
        for ni in range(ns):
            mn = mi * ns + ni
            plt.subplot(ms, ns, mn + 1)
            plt.imshow(imgs[mn, :, :, 1].squeeze())
            plt.axis('off')
            # plt.title('%d, %s, %.1f' % (labels_xy[mi*ns+ni, :], str(np.where(labels_z[mi*ns+ni, :] != 0)[0].tolist()), pred[mi*ns+ni, :]))
            plt.title('%d, %.1f' % (labels_xy[mi*ns+ni, :], preds[mi*ns+ni, :]))
    plt.draw()

    acc = sum(np.multiply(preds, labels_xy) >= 0) / sampleN_test
    figFile = cacheDir + '/fig' + preName + '_acc%.3f' % acc + '.png'
    fig.savefig(figFile)
    plt.pause(5)
    plt.close()

    sess.close()
