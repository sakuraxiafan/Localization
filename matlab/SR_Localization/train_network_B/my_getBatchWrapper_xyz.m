function fn = my_getBatchWrapper_xyz(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) my_getBatch(imdb,batch,opts) ;

function y = my_getBatch(imdb, images, varargin)
% GET_BATCH  Load, preprocess, and pack images for CNN evaluation

opts.outputRange = [-1 1];

opts.useGpu = false ;
opts.normFlag = false;
opts.outputSize = [1, 1, 100];
opts.imageSize = [64, 64, 64] ;
opts.predSize = opts.imageSize;

opts.transformation = 'none' ;
opts.numThreads = 1 ;
[opts,~] = vl_argparse(opts, varargin);

ims = zeros([opts.imageSize, numel(images)], 'single') ;
labelsZ = zeros([opts.outputSize(1:2), opts.outputSize(3), numel(images)], 'single') ;
labelsX = zeros([opts.outputSize(1:2), 1, numel(images)], 'single') ;
labelsY = zeros([opts.outputSize(1:2), 1, numel(images)], 'single') ;

for i=1:numel(images)
    idx = images(i);
    ims(:,:,:,i) = imdb.imstack(:,:,idx);
    labelZ = round((imdb.ground_truth.z(idx)-opts.outputRange(1,1))/...
        (opts.outputRange(1,2)-opts.outputRange(1,1))*(opts.outputSize(3)-1)+1);
    labelsZ(:,:,:,i) = -1; labelsZ(:,:,labelZ,i) = 1;
    labelX = round((imdb.ground_truth.x(idx)-opts.outputRange(2,1))/...
        (opts.outputRange(2,2)-opts.outputRange(2,1))*(opts.outputSize(3)-1)+1);
    labelsX(:,:,:,i) =  labelX;
    labelY = round((imdb.ground_truth.y(idx)-opts.outputRange(3,1))/...
        (opts.outputRange(3,2)-opts.outputRange(3,1))*(opts.outputSize(3)-1)+1);
    labelsY(:,:,:,i) =  labelY;
end

if opts.useGpu
    ims = gpuArray(ims);
end

y = {'data', ims, 'labelZ', labelsZ, 'labelX', labelsX, 'labelY', labelsY} ;
