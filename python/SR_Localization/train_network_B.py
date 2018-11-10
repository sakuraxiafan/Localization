from keras.models import Model
from keras.layers import Dense, GlobalMaxPool2D, Dropout, Conv2D, MaxPooling2D
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
expType, expNo = 'B', '01'
net_continue = 0

nb_epoch, batch_size = 5, 8
sampleN = int(2e5)
imageSize = [13, 13, 3]
outputSize = [1, 100, 100, 100]
show_samples = False
lr, momentum, decay = 1e-3, 0.9, 5e-4
plt.ion()

##############################
#           set GPU
##############################
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3333)

for model_type, model_name in model_types.items():
    print('\n##############################\n Run for model: %s\n##############################' % model_type)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    ##############################
    #           get model
    ##############################
    cacheDir = dataPath + '/SR_' + expType + expNo
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

        img_input = Input(shape=imageSize)
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        x = GlobalMaxPool2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        pred_x = Dense(outputSize[1], activation='softmax', name='out_x')(x)
        pred_y = Dense(outputSize[2], activation='softmax', name='out_y')(x)
        pred_z = Dense(outputSize[3], activation='softmax', name='out_z')(x)
        model = Model(inputs=img_input, outputs=[pred_x, pred_y, pred_z])
        model.compile(optimizer=SGD(lr=lr, momentum=momentum, decay=decay),
                      loss={'out_x': 'categorical_crossentropy',
                            'out_y': 'categorical_crossentropy',
                            'out_z': 'categorical_crossentropy'},
                      loss_weights=[1., 1., 1.], metrics=['categorical_accuracy'])

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
    labels_xy, preds_xy = np.zeros([data_num, outputSize[0]]), np.zeros([data_num, outputSize[0]])
    labels_x, preds_x = np.zeros([data_num, outputSize[1]]), np.zeros([data_num, outputSize[1]])
    labels_y, preds_y = np.zeros([data_num, outputSize[2]]), np.zeros([data_num, outputSize[2]])
    labels_z, preds_z = np.zeros([data_num, outputSize[3]]), np.zeros([data_num, outputSize[3]])

    for i in range(int(data_num / batch_size)):
        start_num, end_num = i * batch_size, (i + 1) * batch_size
        if i == int(data_num / batch_size)-1:
            end_num = data_num
        img, label_xy, label_x, label_y, label_z = test_gen.generate_batch(start_num=start_num, end_num=end_num)
        pred = model.predict_on_batch(img)

        imgs[start_num:end_num, :, :, :] = img
        labels_xy[start_num:end_num, :] = label_xy
        labels_x[start_num:end_num, :] = label_x
        labels_y[start_num:end_num, :] = label_y
        labels_z[start_num:end_num, :] = label_z

        preds_x[start_num:end_num, :] = pred[0]
        preds_y[start_num:end_num, :] = pred[1]
        preds_z[start_num:end_num, :] = pred[2]

    ms, ns = 2, 5
    fig = plt.figure()
    for mi in range(ms):
        for ni in range(ns):
            mn = mi * ns + ni
            plt.subplot(ms*2, ns, mi*2 * ns + ni + 1)
            plt.imshow(imgs[mn, :, :, 1].squeeze())
            plt.axis('off')
            plt.subplot(ms*2, ns, mi*2 * ns + ni + 1 + ns)
            plt.plot(preds_x[mi*ns+ni, :]/np.max(preds_x[mi*ns+ni, :]), 'g-')
            plt.plot(labels_x[mi*ns+ni, :]/np.max(labels_x[mi*ns+ni, :]), 'g-.')
            plt.plot(preds_y[mi*ns+ni, :]/np.max(preds_y[mi*ns+ni, :]), 'b-')
            plt.plot(labels_y[mi*ns+ni, :]/np.max(labels_y[mi*ns+ni, :]), 'b-.')
            plt.plot(preds_z[mi*ns+ni, :]/np.max(preds_z[mi*ns+ni, :]), 'r-')
            plt.plot(labels_z[mi*ns+ni, :]/np.max(labels_z[mi*ns+ni, :]), 'r-.')
    plt.draw()

    acc = np.mean([sum(np.argmax(preds_x, -1) == np.argmax(labels_x, -1)) / sampleN_test,
                   sum(np.argmax(preds_y, -1) == np.argmax(labels_y, -1)) / sampleN_test,
                   sum(np.argmax(preds_z, -1) == np.argmax(labels_z, -1)) / sampleN_test])
    figFile = cacheDir + '/fig' + preName + '_acc%.3f' % acc + '.png'
    fig.savefig(figFile)
    plt.pause(10)
    plt.close()

    sess.close()
