clear; close all;

run ../dependencies/matconvnet-1.0-beta24/matlab/vl_setupnn;
addpath ../dependencies/matconvnet-1.0-beta24/examples;
addpath ../my_util;

addpath('test_network');

rootDir = '../..';
dataPath = [rootDir filesep 'data']; my_mkdir(dataPath);
opts.cal=load([dataPath filesep 'exp_psf' filesep 'bead_astig_3dcal.mat']);

if ispc, opts.gpus = []; else, opts.gpus = 2; end
if opts.gpus, gpuDevice(opts.gpus); end

xyscales = 132;

opts.expNo = 'A01';
opts.expName = 'SR_';
opts.modelType = [opts.expName opts.expNo];
opts.expDir = [dataPath filesep opts.modelType];
list = dir(fullfile(opts.expDir, 'net-epoch-*.mat')) ;
load([opts.expDir filesep list(end).name], 'net');
outmodel = dagnn.DagNN.loadobj(net) ;
if opts.gpus, outmodel.move('gpu'); end
bopts.imageSize = [13, 13, 1];
topts.windowSize = bopts.imageSize(1:2);
topts.windowStride = [1, 1];
topts.outs = [outmodel.getVarIndex('prediction'), -1];
topts.imageSize = bopts.imageSize;
topts.useGpu = opts.gpus;

expNos = {'B01'};
outs = [];
for expI = 1:length(expNos)
    opts.expNo = expNos{expI};
    opts.expName = 'SR_';
    opts.modelType = [opts.expName opts.expNo];
    opts.expDir = [dataPath filesep opts.modelType];
    list = dir(fullfile(opts.expDir, 'net-epoch-*.mat')) ;
    load([opts.expDir filesep list(end).name], 'net');
    net = dagnn.DagNN.loadobj(net) ;
    if opts.gpus, net.move('gpu'); end
    
    if expI==1, nets = net; else, nets(expI) = net; end
    outs(end+1) = net.getVarIndex('predictionX');
    outs(end+1) = net.getVarIndex('predictionY');
    outs(end+1) = net.getVarIndex('predictionZ');
end

cal=load([dataPath filesep 'exp_psf' filesep 'bead_astig_3dcal.mat']);

expNo = 'v01';

numlocs = 100;
densities = [1 2 4 8 16 32];
showImgs = [1 1];
tic
imgsWhole = {}; imstacks = {}; dds = {}; gts = {}; cnnPrds = {};
cnnRes = struct('tp',0,'fp',0,'fn',0,'mae',0,'rmse',0,'pr',0,'rc',0,'f1',0,'mdae',0,'jac',0);
cMatchD = 200; allFlag = 1;
for densityI = 1:length(densities)
    density = densities(densityI);

    rng(numlocs);
    Intensity=2000; %number of photons / localization
    background=10; %number of background photons per pixel
    
    Pixelsize = 64;
    Psize = 13;
    RoiPixelsize = Pixelsize - Psize; %ROI in pixels for simulation
    dz=cal.cspline.dz;  %coordinate system of spline PSF is corner based and in units pixels / planes
    z0=cal.cspline.z0; % distance and midpoint of stack in spline PSF, needed to translate into nm coordinates
    dx=floor(Pixelsize/2); %distance of center from edge
    zRange = [-1 1]*500;
    
    coordinates = [];
    for i=1:density
        gt.x = (rand(numlocs,1)*2-1)*RoiPixelsize/2;
        gt.y = (rand(numlocs,1)*2-1)*RoiPixelsize/2;
        gt.z = rand(numlocs,1)*(zRange(2)-zRange(1))+zRange(1);
        coordinates = horzcat(coordinates,horzcat(gt.x+dx,gt.y+dx,gt.z/dz+z0));
    end
    rng(numlocs);
    imgs = my_simSplinePSF(Pixelsize,cal.cspline.coeff,Intensity,background,coordinates); %simulate images
    
    for k = 1:size(imgs,3)
        img = imgs(:,:,k);
        
        
        pred = my_fun_1_evaluateSingle({outmodel,[]}, img, topts, 1);
        mask = pred>=0 & imdilate(pred, ones(3))==pred;
        [xx, yy] = find(mask);
        dxx = xx+floor(topts.windowSize(1)/2);
        dyy = yy+floor(topts.windowSize(2)/2);

        
        Prange = [(Psize-1)/2 + 1, Pixelsize - (Psize-1)/2];
        dxx(dxx<Prange(1)) = Prange(1); dxx(dxx>Prange(2)) = Prange(2);
        dyy(dyy<Prange(1)) = Prange(1); dyy(dyy>Prange(2)) = Prange(2);
        
        imstack = []; dd = [];
        for m = 1:length(dxx)
            patch = img(dxx(m)+((1-Psize)/2:(Psize-1)/2), dyy(m)+((1-Psize)/2:(Psize-1)/2)); % my_imagesc(patch);
            imstack = cat(3, imstack, patch);
            dd = cat(1, dd, [dxx(m), dyy(m)]);
        end
        
        preds = zeros(size(imstack,3), 3);
        batchSize = 20;
        for i = 1:ceil(size(imstack,3)/batchSize)
            
            if i == ceil(size(imstack,3)/batchSize), ii = (i-1)*batchSize+1 : size(imstack,3);
            else, ii = (i-1)*batchSize+1 : i*batchSize;
            end
            
            im = single(imstack(:,:,ii));
            im = reshape(im, [size(im,1) size(im,2), 1, size(im,3)]);
            if opts.gpus
                im = gpuArray(im);
            end
            
            for expI = 1:length(expNos)
                inputs = {'data', im, 'labelZ', [], 'labelX', [], 'labelY', []};
                nets(expI).eval(inputs) ;
                
                for jj=1:3
                    pred = gather(nets(expI).vars(outs(jj)).value);
                    
                    pred_tmp = exp(squeeze(pred));
                    pred_tmp = bsxfun(@rdivide,pred_tmp,sum(pred_tmp,1));
                    
                    bopts.outputSize = [1, 1, 100];
                    if jj~=3, bopts.outputRange = [-1 1]*1; else, bopts.outputRange = [-1 1]*500; end
                    label_tmp = bopts.outputRange(1):(bopts.outputRange(2)-bopts.outputRange(1))/...
                        (bopts.outputSize(3)-1):bopts.outputRange(2);
                    pred_final = (label_tmp*pred_tmp)';
                    
                    preds(ii, jj) = pred_final;
                end
            end
        end
        
        dx = floor(size(imstack,1)/2);
        tmpRes.x=(preds(:,1)+dd(:,1)-1.5)*xyscales;
        tmpRes.y=(preds(:,2)+dd(:,2)-1.5)*xyscales;
        tmpRes.z=preds(:,3);
        
        
        centers = [tmpRes.x, tmpRes.y, tmpRes.z];
        gtc = reshape(coordinates(k,:), [3, density])';
        gtc(:,1:2) = gtc(:,1:2)*xyscales;
        gtc(:,3) = (gtc(:,3)-cal.cspline.z0)*cal.cspline.dz;
        [res, pairs] = my_evalRes([],gtc,centers,cMatchD,allFlag);
        
        if showImgs(1)
            figure(1), imagesc(img); hold on
            plot(gtc(:,2)/xyscales+1.5,gtc(:,1)/xyscales+1.5,'r.');
            plot(centers(:,2)/xyscales+1.5,centers(:,1)/xyscales+1.5,'r^');
            pause(1);
        end
        
        if showImgs(2)
            ms = 3; ns = 5;
            figure(2), set(gcf,'Position',[100,100,1400,700]);
            for i=1:ms
                for j=1:ns
                    kk = (i-1)*ns+j; if size(imstack,3)<kk; break; end
                    subplot(ms,ns,kk), imagesc(imstack(:,:,kk)); hold on;
                    plot(gtc(:,2)/xyscales+2.5-dd(kk,2)+dx,gtc(:,1)/xyscales+2.5-dd(kk,1)+dx,'r.');
                    plot(centers(:,2)/xyscales+2.5-dd(kk,2)+dx,centers(:,1)/xyscales+2.5-dd(kk,1)+dx,'r^');
                end
            end
            pause(1);
        end
        
        imgsWhole{densityI, k} = img;
        imstacks{densityI, k} = imstack;
        dds{densityI, k} = dd;
        gts{densityI, k} = gtc;
        
        cnnPrds{densityI, k} = centers;
        cnnRes(densityI, k) = res;
    end
    
    rmse = []; jpr = [];
    for k = 1:size(imgs,3)
        rmse = cat(1, rmse, cnnRes(densityI,k).rmse);
        jpr = cat(1, jpr, [cnnRes(densityI,k).jac cnnRes(densityI,k).pr cnnRes(densityI,k).rc]);
    end
    rmse_avg = mean((sum(rmse.^2,2)).^.5);
    rmse = (sum(rmse.^2,1)/size(rmse,1)).^.5;
    fprintf('density %.2f: \trmse -\t%.1f \t(%.1f,%.1f,%.1f) \tjac -\t%.3f (%.3f,%.3f)\n',density,...
        rmse_avg,rmse(1),rmse(2),rmse(3),mean(jpr(:,1)),mean(jpr(:,2)),mean(jpr(:,3)))
end
toc
