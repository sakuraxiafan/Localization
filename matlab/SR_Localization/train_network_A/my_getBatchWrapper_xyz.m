function fn = my_getBatchWrapper_xyz(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) my_getBatch(imdb,batch,opts) ;

function y = my_getBatch(imdb, images, varargin)
% GET_BATCH  Load, preprocess, and pack images for CNN evaluation

opts.outputRange = [-1 1];
opts.cPixel = 2;

opts.useGpu = false ;
opts.normFlag = false;
opts.outputSize = [1, 1, 100];
opts.imageSize = [64, 64, 64] ;
opts.predSize = opts.imageSize;

opts.transformation = 'none' ;
opts.numThreads = 1 ;
[opts,~] = vl_argparse(opts, varargin);

ims = zeros([opts.imageSize, numel(images)], 'single') ;
labels = zeros([opts.outputSize(1:2), 1, numel(images)], 'single') ;

for i=1:numel(images)
    idx = images(i);
    ims(:,:,:,i) = imdb.imstack(:,:,idx);
    xs = imdb.gt_bg.x(idx,:); ys = imdb.gt_bg.y(idx,:);
    zeroMask = xs~=0 & ys~=0; xs = xs(zeroMask); ys = ys(zeroMask);
    labels(:,:,:,i) = (any(xs.^2+ys.^2 <= opts.cPixel^2)-0.5)*2;
end

if opts.useGpu
    ims = gpuArray(ims);
end

y = {'data', ims, 'label', labels} ;

% mi=3; ni=4;
% for tmp=1:mi*ni
%     if tmp>size(ims,4), continue; end
%     subplot(mi,ni,tmp),imshow(ims(:,:,:,tmp)/100);title(sprintf('gt:%d',labels(:,:,:,tmp)));
% end
% pause;

