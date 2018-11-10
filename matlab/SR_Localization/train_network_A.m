clear; close all;

run ../dependencies/matconvnet-1.0-beta24/matlab/vl_setupnn;
addpath ../dependencies/matconvnet-1.0-beta24/examples;
addpath ../my_util;

addpath('train_network_A');

rootDir = '../..';
dataPath = [rootDir filesep 'data']; my_mkdir(dataPath);
opts.cal=load([dataPath filesep 'exp_psf' filesep 'bead_astig_3dcal.mat']);

opts.expNo = 'A01';
opts.expName = 'SR_';
opts.sampleN = 2e5;

testMode = 0; % 0-training; 1-code examing; 2-run examples

if ispc, opts.train.gpus = []; else, opts.train.gpus = 1; end
opts.train.derOutputs = {'loss',1e0};
lrN = 10;
trainOpts.learningRate = 1e-3*[ones(1,lrN/2) 0.1*ones(1,lrN/2)];
trainOpts.continue = true; % false; % 
trainOpts.saveEach = true; % false; % 

bopts.imageSize = [13, 13, 1];
bopts.outputSize = [1, 1, 1];
bopts.numAugments = 1;
bopts.Intensity = 2000;
bopts.background = 10;
bopts.outputRange = [-1 1]*1;
bopts.xyzScale = 1/132;
opts.predMeth = 'probability'; % 'probability' | 'max'

bopts.density = [0 1 1 2 2 2 3 3 3 4 4 5];
bopts.bgRange = 2;
bopts.fixXY = 0.5;
bopts.cPixel = 1;

trainOpts.batchSize = 64;
trainOpts.numSubBatches = 8;

% path setup
opts.modelType = [opts.expName opts.expNo];
opts.expDir = [dataPath filesep opts.modelType];
opts.logFile = [opts.expDir filesep 'log-' opts.modelType '.txt'];
opts.imgFile = [opts.expDir filesep 'img-' opts.modelType '.bmp'];
opts.model = [];

trainOpts.prefetch = false;
trainOpts.expDir = opts.expDir;
trainOpts.numEpochs = numel(trainOpts.learningRate);

opts.imdbPath = [dataPath filesep 'imdb' filesep 'imdb' ...
    sprintf('_n%d_d%d %d_r%.1f_f%.1f.mat',...
    opts.sampleN,min(bopts.density),max(bopts.density),bopts.bgRange,bopts.fixXY)];
if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath);
else
    imdb = my_setup_sm(opts, bopts);
    my_mkdir(fileparts(opts.imdbPath));
    save(opts.imdbPath, '-struct', 'imdb');
end

net = my_cnnInitializeModel_xyz(bopts, 'model', opts.model);

net.meta.normalization.rgbMean = 0 ;
net.meta.classes = imdb.classes.name ;

rng(opts.sampleN);
imgN = size(imdb.imstack,3);
ratio = [0.98, 0.01, 0.01];
imgNs = round(imgN*ratio);
randList = randperm(imgN);
train = randList(1:imgNs(1));
val = randList(imgNs(1)+(1:imgNs(2)));
test = randList(imgNs(1)+imgNs(2)+(1:imgNs(3)));

bopts.useGpu = opts.train.gpus;
var_sizes = net.getVarSizes({'data', bopts.imageSize})';

if testMode>=1, train = train(1); val = val(1); trainOpts.numEpochs = 1; end

if opts.train.gpus, gpuDevice(opts.train.gpus); end

if testMode<=1
info = cnn_train_dag(net, imdb, my_getBatchWrapper_xyz(bopts), ...
                     trainOpts, ....
                     'train', train(1:end), ...
                     'val', val(1:end), ...
                     opts.train) ;
end

netOpts.mode = 'test'; % 'normal' better than 'test' while testing
netOpts.plot = false;
netOpts.testBatch = trainOpts.batchSize;
if isempty(opts.train.gpus), netOpts.gpu = false;
else, netOpts.gpu = opts.train.gpus;     % set to true to enable GPU support
end
if exist('info','var'), outmodel = info; else, outmodel = net; end



