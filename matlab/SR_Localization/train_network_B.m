clear; close all;

run ../dependencies/matconvnet-1.0-beta24/matlab/vl_setupnn;
addpath ../dependencies/matconvnet-1.0-beta24/examples;
addpath ../my_util;

addpath('train_network_B');

rootDir = '../..';
dataPath = [rootDir filesep 'data']; my_mkdir(dataPath);
opts.cal=load([dataPath filesep 'exp_psf' filesep 'bead_astig_3dcal.mat']);

opts.expNo = 'B01';
opts.expName = 'SR_';
opts.sampleN = 2e5;

testMode = 0; % 0-training; 1-code examing; 2-run examples; 3-test outputs

if ispc, opts.train.gpus = []; else, opts.train.gpus = 1; end
opts.train.derOutputs = {'lossZ',1e0,'lossX',1e0,'lossY',1e0};
lrN = 10;
trainOpts.learningRate = 1e-3*[ones(1,lrN/2) 0.1*ones(1,lrN/2)];
trainOpts.continue = false;
trainOpts.saveEach = true;

bopts.imageSize = [13, 13, 1];
bopts.outputSize = [1, 1, 100];
bopts.numAugments = 1;
bopts.Intensity = 2000;
bopts.background = 10;
bopts.outputRange = [[-1 1]*500; [-1 1]*1; [-1 1]*1];
bopts.xyzScale = [1 1/132 1/132];
opts.predMeth = 'probability'; % 'probability' | 'max'

bopts.density = 2;
bopts.bgRange = 1.5;

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

opts.imdbPath = [dataPath filesep 'imdb' filesep 'imdb' sprintf('_n%d',opts.sampleN)...
                 sprintf('_d%d',bopts.density) sprintf('_r%.1f',bopts.bgRange) '.mat'];
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

% rng(opts.neuroN);
imgN = size(imdb.imstack,3);
ratio = [0.98, 0.01, 0.01];
imgNs = round(imgN*ratio);
randList = randperm(imgN);
train = randList(1:imgNs(1));
val = randList(imgNs(1)+(1:imgNs(2)));
test = randList(imgNs(1)+imgNs(2)+(1:imgNs(3))); test = sort(test);

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
else, netOpts.gpu = opts.train.gpus;
end
if exist('info','var'), outmodel = info; else, outmodel = net; end


opts.sampleN = 5e3;
opts.imdbPath = [dataPath filesep 'imdb' filesep 'imdb' sprintf('_n%d',opts.sampleN)...
                 sprintf('_d%d',bopts.density) sprintf('_r%.1f',bopts.bgRange) '.mat'];
if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath);
else
    imdb = my_setup_sm(opts, bopts);
    my_mkdir(opts.expDir);
    save(opts.imdbPath, '-struct', 'imdb');
end
test = 1:opts.sampleN;

ms = 4; ns = 6; testN = min(length(test),ms*ns/2);
testIdx = 1:floor(length(test)/testN):length(test);
[img, pred, gt] = my_prediction_xyz(outmodel, ... net, ... 
    imdb, my_getBatchWrapper_xyz(bopts), test(testIdx), opts, netOpts);
figure, set(gcf,'Position',[100,100,1400,700]);
for i=1:ms/2
    for j=1:ns
        k = testIdx(i*ns+j-ns); kk = 2*(i-1)*ns+j;
        subplot(ms,ns,kk), imagesc(imdb.imstack(:,:,k));
        title(sprintf('gt:(%.2f,%.2f,%.0f)',...
            imdb.ground_truth.x(k),imdb.ground_truth.y(k),imdb.ground_truth.z(k)));
        subplot(ms,ns,kk+ns), hold on;
        plot(squeeze(pred{2}(:,:,:,i*ns+j-ns)),'g');
        plot(squeeze(pred{3}(:,:,:,i*ns+j-ns)),'b');
        plot(squeeze(pred{1}(:,:,:,i*ns+j-ns)),'r');
        title(sprintf('bg:(%.2f,%.2f,%.0f)',...
            imdb.gt_bg.x(k,1),imdb.gt_bg.y(k,1),imdb.gt_bg.z(k,1)));
    end
end

