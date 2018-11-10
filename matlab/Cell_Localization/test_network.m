clear; % close all;

run ../dependencies/matconvnet-1.0-beta24/matlab/vl_setupnn;
addpath ../dependencies/matconvnet-1.0-beta24/examples;
addpath ../my_util;

addpath('test_network');

rootDir = '../..';
dataPath = [rootDir filesep 'data']; my_mkdir(dataPath);

if ispc, opts.train.gpus = []; else, opts.train.gpus = 2; end;
if opts.train.gpus, gpuDevice(opts.train.gpus); end;


rePred = 0;

expN = 'Test01';
saveDir = [dataPath filesep 'Cell_' expN]; my_mkdir(saveDir);
videoDir = [dataPath filesep 'videos'];

expNoss = {{'Cell_A01','Cell_B01'}};

if ispc, opts.useGpu = []; else, opts.useGpu = 1; end

deltaFrame = 1; imgNMax = Inf;

% set params
thr_xy = -1; 
filterSize = 0; cMatchD = 10; allFlag = 1;
opts.normFlag = 1;
opts.windowSize = 128*ones(1,2);
XYum = 250;
opts.imageSize = [XYum*2*ones(1,2)+opts.windowSize 3];
opts.windowStride = 4*ones(1,2);
opts.areaL = opts.imageSize(1:2)-opts.windowSize;
opts.r = 0;

psfMaxZ = 50; dz = 1;
dispScale = 1*[1 1 1].*[1/2./opts.areaL 1/8/psfMaxZ];

viewPoint = [-20,30];
ta = 170; tb = 360;

if opts.useGpu, gpuDevice(opts.useGpu); end

ress = cell(0);
for expNosI = 1:length(expNoss)
    expNos = expNoss{expNosI};
    
    saveForm = 'png';

    if length(expNos)==2
        nets = cell(length(expNos),1); outs = zeros(size(nets));
        for expNoI = 1:length(expNos)
            opts.expNo = expNos{expNoI};
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

    bin = 2;
    imgR = [50 100 200 250];
    
    imgRtmp = [imgR(1:2)+imgR(3:4)/2 imgR(3:4)];
    scaleTmp = 1; imgR = round([imgRtmp(1:2)-imgRtmp(3:4)/2/scaleTmp,imgRtmp(3:4)/scaleTmp]);
    
    imgR(3:4) = imgR(1:2)+imgR(3:4); imgR = imgR([2 1 4 3]);
    
    
    imgS = imgR(3:4)*bin-imgR(1:2)*bin+1;
    
    Nz = psfMaxZ*2;
    
    iSz = imgS;
    wSz = opts.windowSize;
    wSt = opts.windowStride;
    roi = [[floor(wSz/2) 0]+1; [floor(wSz/2)+floor((iSz-wSz)./wSt).*wSt Nz/2]-opts.r]; roi=roi(:)';
    opts.roi = roi;

    
    imgList = my_dir(videoDir,'*.tif');
    imgInfo = imfinfo(imgList{1});
    imgN = length(imgInfo);
    
    preName = sprintf('fig_%s',[expNos{:}]);
    
    rng(0); res = []; imgs = zeros(imgR(3)-imgR(1)+1,imgR(4)-imgR(2)+1,imgN);
    imgIs = 1:deltaFrame:imgN;
    frameN = min(length(imgIs),imgNMax);
    
    for kI=1:imgN
        progress('Reading imgs',kI,imgN);
        img = double(imread(imgList{1},kI,'PixelRegion',{[imgR(1),imgR(3)], [imgR(2),imgR(4)]}));
        imgs(:,:,kI) = img;
    end
    imgs = imresize(imgs,1\bin);
    
    showKs = ta:tb;
    for kI=showKs
        
        fprintf('Calc img %03d of %03d\n', kI, frameN);
        
        img = imgs(:,:,imgIs(kI));
        img = img-min(imgs,[],3);
        
        
        preds_z = []; preds_xy = [];
        for expNoI = 1:length(expNos)
            tmpFile = [saveDir filesep sprintf('%s_F%03d.mat',expNos{expNoI},imgIs(kI))];
            if exist(tmpFile,'file') && ~rePred
                load(tmpFile,'img','opts','preds');
            else
                preds = my_fun_1_evaluateSingle(nets, img, opts, expNoI);
                save(tmpFile,'img','opts','preds');
            end
            if expNoI==2, preds_z = preds; else, preds_xy = preds; end
            
        end
        
        preds_z_org = preds_z;
        preds_z = bsxfun(@rdivide,exp(preds_z),sum(exp(preds_z),3));
        predSz = size(preds_z)-1; roiSz = roi([2 4 6])-roi([1 3 5]);
        [predMgx, predMgy] = meshgrid((0:predSz(1))/predSz(1),(0:predSz(2))/predSz(2));
        [roiMgx, roiMgy] = meshgrid((0:roiSz(1))/roiSz(1),(0:roiSz(2))/roiSz(2));
        preds_xy_scale = interp2(predMgx, predMgy, preds_xy', roiMgx, roiMgy)';
        [predMgx, predMgy, predMgz] = meshgrid((0:predSz(1))/predSz(1),(0:predSz(2))/predSz(2),(0:predSz(3))/predSz(3));
        [roiMgx, roiMgy, roiMgz] = meshgrid((0:roiSz(1))/roiSz(1),(0:roiSz(2))/roiSz(2),(0:roiSz(3))/roiSz(3));
        preds_z_scale = permute(interp3(predMgx,predMgy,predMgz,permute(preds_z,[2 1 3]),roiMgx,roiMgy,roiMgz),[2,1,3]);
        
        centers = my_fun_2_locationFit(preds_xy_scale, thr_xy,filterSize,[],10);
        [~,maxZ] = max(preds_z_scale,[],3);
        maxZ = imerode(maxZ,strel('diamond',1));
        centers(:,3) = maxZ(sub2ind(size(maxZ),centers(:,1),centers(:,2)));
        
        
        centers(centers(:,3)>=47,:) = []; % add-0313
        
        res(end+1).centers = centers;
        
        fprintf('Find %d points.\n', size(centers,1));
        
    end
    ress{expNosI} = res;
end


cutRate = 0.998;

rng(5); colorsTrack1 = rand(6,3);
colorsTrack = [colorsTrack1; 1,0.4,1; 0.2,0.9,0.3];
colorsIdx = [1 3 2 4 5 6 7 8; 1 2 5 4 3 7 6 8];
predTsAll = cell(0);
for expNosI = 1:length(expNoss)
    expNos = expNoss{expNosI};
    preName = sprintf('fig_%s_D%03d_%s%s',[expNos{:}]);
    colorTrack = colorsTrack(colorsIdx(expNosI,:),:);


expIs = [1];
centersList = [];
for ii = 1:length(expIs)
for f=1:length(showKs)
    centers = ress{expNosI}(f).centers;
    centersList = [centersList; [centers f*ones(size(centers,1),1)]];
end
end

cutRate = 0.998;

scales = [1 1 5];
centersList_tmp = centersList;
centersList_tmp(:,1:3) = bsxfun(@times,centersList(:,1:3),scales);
try
    paraT.costOfNonAssignment = 30;
    paraT.invisibleForTooLong = 5;
    paraT.ageThreshold = 10;
    paraT.iniError = [20,5];
    paraT.MotNoise = [1,0.05]*1.5;
    paraT.MeaNoise = 10/3;
    predTs = my_kalmanTracking(centersList_tmp,paraT);
% catch, continue;
end
predTs(:,1:3) = bsxfun(@rdivide,predTs(:,1:3),scales);
dur = [];
for pi = unique(predTs(:,end))'
    predT = predTs((predTs(:,end)==pi),:);
    dur(pi) = size(predT,1);
end
[~,idx] = sort(dur,'descend');
predTs_tmp = []; ii = 1;
for pi = idx(1:min(length(idx),8))
    tmp = predTs((predTs(:,end)==pi),:);
    tmp(:,end) = ii; ii=ii+1;
    predTs_tmp = cat(1,predTs_tmp,tmp);
end
predTs = predTs_tmp;
predTsAll{end+1} = predTs;

% 3d comp
figure('Color',[1 1 1]), set(gcf,'Position',[100,100,800,700]);
shapesL = {'-','-.'}; linesz = [1 3]; shapesP = {'.','s'}; marksz = [15 5];fontSz = 20;
colors = [1 0 0; 0 1 0]; 
grid on; hold on; view(viewPoint); set(gca,'YDir','reverse');
for i=unique(predTs(:,end))'
    predT = predTs((predTs(:,end)==i),:);
    for f=1:size(predT,1)-1
        plot3(predT(f:f+1,1),predT(f:f+1,2),predT(f:f+1,3),shapesL{1},...
            'Color',colorTrack(i,:),'LineWidth',linesz(2));
    end
end
xlim([0,iSz(1)-wSz(1)]); ylim([0,iSz(2)-wSz(2)]); zlim(roi(5:6)-[1 0]);
set(gca,'XTick',[0,XYum/5*2,XYum/5*4,XYum/5*6,XYum/5*8,XYum*2],'XTickLabel',...
    {'0',num2str(XYum/5*1),num2str(XYum/5*2),[num2str(XYum/5*3) ''],num2str(XYum/5*4),num2str(XYum)},'fontsize',fontSz);
yt = (roi(4)-roi(3));
set(gca,'YTick',[yt-XYum/5*4,yt-XYum/5*2,yt],'YTickLabel',...
    {num2str(XYum/5*2),num2str(XYum/5*1),'0'},'fontsize',fontSz);
set(gca,'ZTick',[0,10,20,30,40,50],'ZTickLabel',{'0','20','40','60','80','100'},'fontsize',fontSz);
xlabel('y [\mum]'); ylabel('x [\mum]');zlabel('z [\mum]');

export_fig(gcf, [saveDir filesep preName '_tracklets' sprintf('_%03d-%03d',ta,tb) '.' saveForm]);
export_fig(gcf, [saveDir filesep preName '_tracklets' sprintf('_%03d-%03d',ta,tb) '.' 'fig']);
end


