clear; % close all;

run ../dependencies/matconvnet-1.0-beta24/matlab/vl_setupnn;
addpath ../dependencies/matconvnet-1.0-beta24/examples;
addpath ../my_util;

addpath('test_network');

rootDir = '../..';
dataPath = [rootDir filesep 'data']; my_mkdir(dataPath);

if ispc, opts.train.gpus = []; else, opts.train.gpus = 2; end;
if opts.train.gpus, gpuDevice(opts.train.gpus); end;

kernelName = 'psf 512_512.tif'; psfFar=1;psfC=26;
opts.dataDir = '';
kernelFile = [dataPath filesep 'exp_psf' filesep kernelName];
kernel = my_readKernel( kernelFile );

psfMaxZ = 50; dz = 1;
if psfFar==0, psfSlice = floor([(psfC-psfMaxZ*dz):dz:psfC (psfC-dz):-dz:(psfC-psfMaxZ*dz)]);
else psfSlice = floor([(psfC+psfMaxZ*dz):-dz:psfC (psfC+dz):dz:(psfC+psfMaxZ*dz)]);
end
kernel = kernel(:,:,psfSlice); % my_showImgStack(kernel,1,0.01);
kernel = kernel - median(kernel(:));

expN = 'Test01';
saveDir = [rootDir filesep 'data' filesep 'Bead_' expN]; my_mkdir(saveDir);
saveForm = 'png';

rePred = 0;
showFigFlag = 0;

% set params
thr_xy = 0; 
cMatchD = 20;
filterSize = 1;
opts.normFlag = 1;
opts.imageSize = [328, 328, 3];
opts.windowSize = 128*ones(1,2);
opts.windowStride = 4*ones(1,2);
opts.areaL = opts.imageSize(1:2)-opts.windowSize;
opts.r = 0;
opts.useGpu = opts.train.gpus;

ks = [1 2 4 8 16 32 64 128]; 
repN = 1:10;

Nz = size(kernel,3);

expNoss = {{'Bead_A01','Bead_B01'}};

for j=1:length(expNoss)
for i=1:length(ks)
    res(i,j,length(repN)) = struct('tp',0,'fp',0,'fn',0,'mae',0,'rmse',0,'pr',0,'rc',0,'f1',0,'mdae',0,'jac',0);
end
end

if showFigFlag, figure(1), clf; set(gcf,'Color',[1 1 1]), set(gcf,'Position',[100,100,500,300]);end;

plotV = [];
for repI = 1:length(repN)

for kI=1:length(ks)

namePro = sprintf('k%02d_r%02d',ks(kI),repN(repI));

% 0. set xyzi
rng(repN(repI)-1);
xyzi = [rand(ks(kI),3)*2-1,rand(ks(kI),1)*0.9+0.1];

Nz = size(kernel,3);
iSz = opts.imageSize(1:2);
wSz = opts.windowSize;
wSt = opts.windowStride;
roi = [[floor(wSz/2) 0]+1; [floor(wSz/2)+floor((iSz-wSz)./wSt).*wSt Nz/2]-opts.r]; roi=roi(:)';
opts.roi = roi;
r = opts.r;

[img,x] = my_sampleGen(xyzi,kernel,opts.normFlag,[],opts.r,...
    [opts.imageSize(1:2),Nz],[opts.areaL(1:2),Nz]);

centersExp = {};
for expNosI = 1:length(expNoss)
    expNos = expNoss{expNosI};
    
    if ~exist('nets','var')
        nets = cell(length(expNos),1); outs = zeros(size(nets));
        for expNoI = 1:length(expNos)
            opts.expNo = expNos{expNoI};
            % load net
            dataPath = [rootDir filesep 'data'];
            opts.modelType = opts.expNo;
            opts.expDir = [dataPath filesep opts.modelType];
            netList = my_dir(opts.expDir,'net-epoch-*.mat',true);
            opts.model = netList{end};
            nets{expNoI} = load(opts.model, 'net');
            nets{expNoI} = dagnn.DagNN.loadobj(nets{expNoI}.net);
            if opts.useGpu, nets{expNoI}.move('gpu'); end
            outs(expNoI) = nets{expNoI}.getVarIndex('prediction');
        end
        opts.outs = outs;
    end
    
    
    preds_z = []; preds_xy = [];
    for expNoI = 1:length(expNos)
        
        tmpFile = [saveDir filesep sprintf('%s_%s_%s.mat',...
            expNos{expNoI},namePro)];
        if exist(tmpFile,'file') && ~rePred
            load(tmpFile,'img','opts','preds','xyzi');
        else
            preds = my_fun_1_evaluateSingle(nets, img, opts, expNoI);
            save(tmpFile,'img','opts','preds','xyzi');
        end
        if expNoI==2, preds_z = preds; else, preds_xy = preds; end
        
    end
    
    preds_z_org = preds_z;
    preds_z = bsxfun(@rdivide,exp(preds_z),sum(exp(preds_z),3));
    predSz = size(preds_z)-1; roiSz = roi([2 4 6])-roi([1 3 5]);
    [predMgx, predMgy] = meshgrid((0:predSz(1))/predSz(1),(0:predSz(2))/predSz(2));
    [roiMgx, roiMgy] = meshgrid((0:roiSz(1))/roiSz(1),(0:roiSz(2))/roiSz(2));
    preds_xy_scale = interp2(predMgx, predMgy, preds_xy, roiMgx, roiMgy);
    [predMgx, predMgy, predMgz] = meshgrid((0:predSz(1))/predSz(1),(0:predSz(2))/predSz(2),(0:predSz(3))/predSz(3));
    [roiMgx, roiMgy, roiMgz] = meshgrid((0:roiSz(1))/roiSz(1),(0:roiSz(2))/roiSz(2),(0:roiSz(3))/roiSz(3));
    preds_z_scale = interp3(predMgx,predMgy,predMgz,preds_z,roiMgx,roiMgy,roiMgz);
    
    
    
    g2c = @(tmp)bsxfun(@rdivide, bsxfun(@minus, tmp, ...
        [size(preds_xy_scale)/2,0]), [opts.areaL(1:2),Nz]/2);
    c2g = @(tmp)abs(bsxfun(@plus, bsxfun(@times, tmp, ...
        [opts.areaL(1:2),Nz]/2), [size(preds_xy_scale)/2,0]));
    
    % 2. predict centers from z values
    centers = my_fun_2_locationFit(preds_xy_scale, thr_xy,1,[],0);
    [~,maxZ] = max(preds_z_scale,[],3);
    centers(:,3) = maxZ(sub2ind(size(maxZ),centers(:,1),centers(:,2)));
    tmpN = size(centers,1);
    
    
    gtc = c2g(xyzi(:,1:3));
    
    res(kI,expNosI,repI) = my_evalRes([],gtc,centers,cMatchD);
    fprintf('%s, maeXY: %.2f, maeZ: %.2f, pr: %.2f, jac: %.2f\n',...
        [expNos{1} '-' namePro],mean(res(kI,expNosI,repI).mae(1:2)),res(kI,expNosI,repI).mae(3),res(kI,expNosI,repI).pr,res(kI,expNosI,repI).jac);
    
    centersExp{expNosI} = centers;
end
end
end

