function fn = my_getBatchWrapper_xyz(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) my_getBatch(imdb,batch,opts) ;

function y = my_getBatch(imdb, images, varargin)
% GET_BATCH  Load, preprocess, and pack images for CNN evaluation

opts.max_d = [16, 16];
opts.r = 2;
opts.normFlag = false;
opts.outputSize = [1, 1, 100];
opts.areaL = 64;
opts.nois = 0;

opts.maxValue = 0;
opts.useGpu = false ;
opts.kernel = [];
opts.imageSize = [64, 64, 64] ;
opts.predSize = opts.imageSize;
opts.numAugments = 1 ;

opts.transformation = 'none' ;
opts.rgbMean = [] ;
opts.rgbVariance = zeros(0,opts.imageSize(3),'single') ;
opts.depMean = [];
opts.labelStride = 1 ;
opts.labelOffset = 0 ;
opts.classWeights = ones(1,21,'single') ;
opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;
opts = vl_argparse(opts, varargin);

kernel = opts.kernel;
imgDepth = opts.imageSize(3);

if ~isempty(opts.rgbVariance) && isempty(opts.rgbMean)
    opts.rgbMean = ones(opts.imageSize(3),1,'single') *0;
end
if ~isempty(opts.rgbMean) && numel(opts.rgbMean)==imgDepth
    opts.rgbMean = reshape(opts.rgbMean, [1 1 imgDepth]) ;
else
    opts.rgbMean = repmat(opts.rgbMean, [1 1 imgDepth]) ;
end

% space for images
ims = zeros([opts.imageSize, numel(images)*opts.numAugments], 'single') ;
% labels_z = zeros([opts.outputSize, numel(images)*opts.numAugments], 'single') ;
labels_z = zeros([opts.outputSize(1:2), 1, numel(images)*opts.numAugments], 'single') ;
labels_xy = zeros([opts.outputSize(1:2), 1, numel(images)*opts.numAugments], 'single') ;

si = 1 ;

for i=1:numel(images)
    
%     xyzi = squeeze(imdb.xyzis(images(i),:,:));
    xyzi = imdb.xyzis{images(i)};
    
%     [img,x] = my_sampleGen(xyzi,kernel,opts.normFlag,...
%         [],opts.r,[opts.imageSize(1:2),size(opts.kernel,3)],...
%         [opts.areaL,opts.areaL,size(opts.kernel,3)],opts.gaus);
    [img,x] = my_sampleGen(xyzi,kernel,opts.normFlag,...
        [],opts.r,[opts.imageSize(1:2)*2,size(opts.kernel,3)],...
        [ones(1,2)*opts.areaL/2^((rand(1)<0.75)*2),size(opts.kernel,3)],opts.imageSize(1:2));
    
%     for nois = [-4 -1],if nois, tmp = imnoise(img,'gaussian',0,10^nois); else tmp = img; end;disp(snr(img,tmp-img)),end;
%     for nois = [-4 -1],if nois, pnois = -12.5-nois; tmp = imnoise(img*10^pnois,'poisson')/10^pnois; else tmp = img; end;disp(snr(img,tmp-img)),end;
    
%     tmp = img;
%     if opts.nois, img = imnoise(img,'gaussian',0,10^opts.nois); end
    if opts.nois, pnois = -12.5-opts.nois; img = imnoise(img*10^pnois,'poisson')/10^pnois; end
%     my_imagesc([tmp,img]);

    ims(:,:,:,si) = repmat(img, [1,1,opts.imageSize(3)]);
    
    
    % vector & multi values
    label_z = -ones(opts.outputSize);
    d = opts.max_d;
    xs = x(floor(end/2+(-d(1)+1:d(1))),floor(end/2+(-d(2)+1:d(2))),:);
    l = find(squeeze(sum(sum(abs(xs),1),2))>1e-7)'; j_begin = 1;
    for j=2:length(l)+1
        if j~=length(l)+1 && l(j)==l(j-1)+1, continue; end;
        l_vs = x(:,:,round((l(j_begin)+l(j-1))/2));
        [l_x,l_y] = find(l_vs>1e-7);
        l_r = min(((l_x-round(size(x,1)/2)).^2+(l_y-round(size(x,2)/2)).^2).^0.5);
%         l_v = 2/(l_r+2);
        l_v = max(-1,min(1,1-l_r/sum(d.^2)^.5*2));
        l_scale = [l(j_begin),l(j-1)]*2-size(x,3)-1;
        l_cur = sort(floor(abs(l_scale)/size(x,3)*opts.outputSize(3)+1));
        if prod(l_scale)<0, l_cur(1) = 1; end;
        label_z(:,:,l_cur(1):l_cur(2))=max(label_z(:,:,l_cur(1):l_cur(2)),l_v);
        j_begin = j;
    end
    label_tmp = squeeze(label_z);
    [~,label_z] = max(label_z);
    label_z = label_z(1);
    labels_z(:,:,:,si) = label_z;
    
    
%     label_xy = any(x(round(end/2),round(end/2),:)>1e-7);
    label_xy = any(label_tmp~=-1);
    if ~label_xy, label_xy = -1; end;
    labels_xy(:,:,:,si) = label_xy;
    
%     set(gcf,'Position',[100,100,600,500]);
%     imagesc(img); title(sprintf('%d',label_xy));
%     pause;
    
    si = si + 1 ;
end
if opts.useGpu
    ims = gpuArray(ims) ;
%     labels = gpuArray(labels) ;
end
% y = {'data', ims, 'label', labels_z, 'label_xy', labels_xy} ;
y = {'data', ims, 'label', labels_xy, 'label_xy', labels_z} ;
% for tmp=1:size(ims,4)
%     subplot(2,4,tmp),imshow(ims(:,:,:,tmp));title(sprintf('gt:%d',labels_xy(:,:,:,tmp)));
%     subplot(2,4,tmp+4),plot(squeeze(labels_z(:,:,:,tmp)));
% end
% pause;