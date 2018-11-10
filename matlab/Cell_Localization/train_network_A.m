clear; % close all;

run ../dependencies/matconvnet-1.0-beta24/matlab/vl_setupnn;
addpath ../dependencies/matconvnet-1.0-beta24/examples;
addpath ../my_util;

addpath('train_network_A');

rootDir = '../..';
dataPath = [rootDir filesep 'data']; my_mkdir(dataPath);

if ispc, opts.train.gpus = []; else, opts.train.gpus = 2; end;
if opts.train.gpus, gpuDevice(opts.train.gpus); end;

kernelName = 'psf cell.tif'; psfFar=1;psfC=1;
opts.dataDir = '';
kernelFile = [dataPath filesep 'exp_psf' filesep kernelName];
kernel = my_readKernel( kernelFile );

psfMaxZ = 50; dz = 1;
if psfFar==0, psfSlice = floor([(psfC-psfMaxZ*dz):dz:psfC (psfC-dz):-dz:(psfC-psfMaxZ*dz)]);
else psfSlice = floor([(psfC+psfMaxZ*dz):-dz:psfC (psfC+dz):dz:(psfC+psfMaxZ*dz)]);
end
kernel = kernel(:,:,psfSlice);

opts.expNo = 'A01';
opts.expName = 'Cell_';
opts.sampleN = 2e5;
opts.neuroN = [1,2];
opts.fixXY = 0.2;
opts.fixI = 0;

bopts.normFlag = 1;
bopts.r = 0;
bopts.max_d = [1 1]*4;
bopts.nois = -2; % snr=3

testMode = 0; % 0-training; 1-code examing; 2-run examples

if ispc, opts.train.gpus = []; else, opts.train.gpus = 2; end;
opts.train.derOutputs = {'loss',1e0};
lrN = 5;
trainOpts.learningRate = 1e-3*ones(1,lrN);
trainOpts.continue = false; % false; % 
trainOpts.saveEach = true; % false; % 

bopts.imageSize = [128, 128, 3];
bopts.outputSize = [1, 1, 1];
bopts.numAugments = 1;
bopts.areaL = min(bopts.imageSize([1 2]))*2;

trainOpts.batchSize = 8;
trainOpts.numSubBatches = 2;

% path setup
opts.modelType = [opts.expName opts.expNo];
opts.expDir = [dataPath filesep opts.modelType];
opts.logFile = [opts.expDir filesep 'log-' opts.modelType '.txt'];
opts.imgFile = [opts.expDir filesep 'img-' opts.modelType '.bmp'];
opts.model = [dataPath filesep 'models' filesep 'imagenet-vgg-verydeep-19.mat'];

% experiment setup
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;

trainOpts.prefetch = false ;
trainOpts.expDir = opts.expDir ;
trainOpts.numEpochs = numel(trainOpts.learningRate) ;

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------

imdb = my_setup_na(opts);
my_mkdir(opts.expDir) ;
save(opts.imdbPath, '-struct', 'imdb') ;

% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------

net = my_cnnInitializeModel_xyz(bopts, 'model', opts.model);

net.meta.normalization.rgbMean = 0 ;
net.meta.classes = imdb.classes.name ;

% rng(opts.neuroN);
imgN = size(imdb.xyzis,1);
ratio = [0.98, 0.01, 0.01];
imgNs = round(imgN*ratio);
randList = randperm(imgN);
train = randList(1:imgNs(1));
val = randList(imgNs(1)+(1:imgNs(2)));
test = randList(imgNs(1)+imgNs(2)+(1:imgNs(3)));

if testMode>=1, train = train(1); val = val(1); trainOpts.numEpochs = 1; end

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------

% Setup data fetching options
bopts.maxValue = 0.5;
bopts.kernel = kernel;
bopts.useGpu = opts.train.gpus;
bopts.labelStride = 1 ;
bopts.labelOffset = 1 ;
bopts.classWeights = ones(1,21,'single') ;
bopts.rgbMean = 0 ;

var_sizes = net.getVarSizes({'data', bopts.imageSize})';
pred_idx = net.getVarIndex('prediction');
pred_size = var_sizes{pred_idx};
bopts.predSize = pred_size(1:3);

if opts.train.gpus, gpuDevice(opts.train.gpus); end;

if testMode~=2
info = cnn_train_dag(net, imdb, my_getBatchWrapper_xyz(bopts), ...
                     trainOpts, ....
                     'train', train(1:end), ...
                     'val', val(1:end), ...
                     opts.train) ;
end
                 

netOpts.mode = 'test'; % 'normal' better than 'test' while testing
netOpts.plot = false;
if isempty(opts.train.gpus), netOpts.gpu = false;
else netOpts.gpu = opts.train.gpus;     % set to true to enable GPU support
end
if exist('info','var'), outmodel = info; else outmodel = net; end;

% for areaLs =  [128 64 40 20 0]
for areaLs =  [min(bopts.imageSize([1 2]))]
bopts.areaL = areaLs;
[img, pred, gt, pred_xy, gt_xy] = my_depthPrediction_xyz(outmodel, ... net, ... 
    imdb, my_getBatchWrapper_xyz(bopts), test(1:16), opts, netOpts);
figure, set(gcf,'Position',[100,100,800,700]);
for i=1:4
    for j=1:4
        k = i*4+j-4;
    subplot(4,4,i*4+j-4), imagesc(img(:,:,1,k));
    title(sprintf('%.2f/%.2f',gt(:,:,:,k),pred(:,:,:,k)));
    end
end
pause(0.01);
end
