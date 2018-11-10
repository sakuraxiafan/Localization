function [ net ] = my_cnnInitializeModel_xyz( varargin )

dilate = 1;

fcRelu = 1;
fcDropout = 0;

removeStride = 0;
addBias = true;
removeBN = 0;
useSigmoid = 0;
removePad = 0;
outputSigmoid = 0;
outputSoftmax = 0;

opts.imageSize = [63, 63, 3];
opts.outputSize = [1, 1, 100];
opts.outputDepth = opts.outputSize(3);
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.lossT = ''; % 'n' | 'w' | 'm' | 'a'
opts.addSumFlag = true;
opts.model = '';
opts.blockN = 5;
% opts.fcs = [1024, 1024];
opts.fcs = [1024];

opts.classNames = {} ;
opts.classDescriptions = {} ;
opts.averageImage = zeros(opts.imageSize(3),1) ;
opts.colorDeviation = zeros(opts.imageSize(3)) ;
opts.cudnnWorkspaceLimit =  640 * 1024^3 ; %1024*1024*1024 ;
[opts, ~] = vl_argparse(opts, varargin) ;

% net = dagnn.DagNN() ;
net = load(opts.model);
net = dagnn.DagNN.fromSimpleNN(net);
net.renameVar('x0','data');

% Modify the bias learning rate for all layers
for i = 1:numel(net.layers)-1
  if (isa(net.layers(i).block, 'dagnn.Conv') && net.layers(i).block.hasBias)
    filt = net.getParamIndex(net.layers(i).params{1}) ;
    bias = net.getParamIndex(net.layers(i).params{2}) ;
    net.params(bias).learningRate = 2 * net.params(filt).learningRate ;
  end
end

net.removeLayer('prob') ;
removeFlag = inf;
poolIdx = [];
for i=1:length(net.layers)
    if strcmp(net.layers(i).name(1:min(end,4)),'pool'), poolIdx(end+1) = i; end
    if strcmp(net.layers(i).name(1:2),'fc'), removeFlag = i; break; end
end
for i=length(net.layers):-1:removeFlag
    net.removeLayer(net.layers(i).name);
end

varSizes = net.getVarSizes({'data',opts.imageSize})';

if removePad
    pads=[];
    for i=1:length(net.layers)
        tmp = findprop(net.layers(i).block,'pad');
        if ~isempty(tmp)
            pads(i,:) = net.layers(i).block.pad;
            net.layers(i).block.pad = zeros(size(net.layers(i).block.pad));
        end
    end
end
varSizes = net.getVarSizes({'data',opts.imageSize})';

if removeStride
    strides = [];
    for i=1:length(net.layers)
        tmp = findprop(net.layers(i).block,'stride');
        if ~isempty(tmp)
            strides(i,:) = net.layers(i).block.stride;
            net.layers(i).block.stride = [1 1];
        end
    end
end

blockN = opts.blockN;
for i = length(net.layers):-1:poolIdx(blockN)+1
    net.removeLayer(net.layers(i).name);
end

if dilate>1
    % add dilate for convolution
    dilates = [];
    for i=1:length(net.layers)
        if ~isa(net.layers(i).block, 'dagnn.Conv'), continue; end;
            dilates(i,:) = net.layers(i).block.dilate;
        net.layers(i).block.dilate = [dilate dilate];
    end
end

oName = net.vars(end).name;
% net.renameVar(net.vars(end).name,oName);

varSizes = net.getVarSizes({'data',opts.imageSize})';

fSize = 1; stride = 1;
numClasses = opts.outputSize(3);

varCurI = str2double(oName(2:end));
% convN = floor(log(varSizes{end}(3)/fDeps(end))/log(4))+1; 
fcN = length(opts.fcs)+1; 
fDeps = [varSizes{end}(3) opts.fcs]; 
for i=1:fcN
        types = {''};
%         types = {'', '_xy'};
    for j = 1:length(types)
        layerT = types{j};        
        varCurT = varCurI;
        
    if i==1
        inputName = sprintf('x%s%d',types{1},varCurT);
    else
        inputName = sprintf('x%s%d',layerT,varCurT);
    end
    
    if i==fcN
        if j==1, fK = numClasses; else, fK = 1; end;
        outputName = sprintf('prediction%s',layerT);
    else
        fK = fDeps(i+1);
        outputName = sprintf('x%s%d',layerT,varCurT+1);
    end
    varCurT = varCurT+1;
    
    if i==1
        sz = [varSizes{end}(1:2), fDeps(i), fK];
    else
        sz = [fSize, fSize, fDeps(i), fK];
    end
    filters = single(randn(sz,'single')* sqrt(2/prod(sz(1:3))));
    if ~addBias
        filterName = sprintf('my_fc%s%df',layerT,i);
    else
        filterName = {sprintf('my_fc%s%df',layerT,i), sprintf('my_fc%s%db',layerT,i)};
    end
    net.addLayer(sprintf('my_fc%s%d',layerT,i), ...
        dagnn.Conv('size', size(filters), ...
        'stride', stride, ....
        'pad', (fSize - 1) / 2, ...
        'hasBias', addBias), ...
        inputName, outputName, filterName) ;
    
    f = net.getParamIndex(sprintf('my_fc%s%df',layerT,i)) ;
    net.params(f).value = filters ;
    net.params(f).learningRate = 1; % 0 ;
    net.params(f).weightDecay = 1 ;
    
    if addBias
        f = net.getParamIndex(sprintf('my_fc%s%db',layerT,i)) ;
        net.params(f).value = zeros([1, 1, fK], 'single') ;
        net.params(f).learningRate = 2; % 0 ;
        net.params(f).weightDecay = 1 ;
    end
    
    
    if fcRelu && i~=fcN
        inputName = sprintf('x%s%d',layerT,varCurT);
        outputName = sprintf('x%s%d',layerT,varCurT+1);
        
        net.addLayer(sprintf('my_fc%s%d_ReLU',layerT,i), dagnn.ReLU(),...
            inputName, outputName);
        varCurT = varCurT+1;
        
        if fcDropout
            inputName = sprintf('x%s%d',layerT,varCurT);
            outputName = sprintf('x%s%d',layerT,varCurT+1);
            
            net.addLayer(sprintf('my_fc%s%dDrouput',layerT,i), dagnn.DropOut(),...
                inputName, outputName);
            varCurT = varCurT+1;
        end
    end
    
    end
    varCurI = varCurT;
end

if removeBN
    % remove bn layers
    for i=length(net.layers):-1:1
        if ~isa(net.layers(i).block, 'dagnn.BatchNorm'), continue; end;
        inputName = net.layers(i).inputs{1};
        outputName = net.layers(i).outputs{1};
        for j=1:length(net.layers)
            for k=1:length(net.layers(j).inputs)
                if strcmp(net.layers(j).inputs{k}, outputName)
                    net.layers(j).inputs{k} = inputName;
                end
            end
        end
        net.removeLayer(net.layers(i).name);
    end
end

if useSigmoid
    % replace relu with sigmoid
    for i=length(net.layers):-1:1
        if ~isa(net.layers(i).block, 'dagnn.ReLU'), continue; end;
        net.layers(i).block = dagnn.Sigmoid;
        net.layers(i).block.attach(net, i) ;
    end
end

if outputSigmoid
    inputName = sprintf('x%s%d',layerT,varCurT);
    outputName = 'prediction';
    net.renameVar(outputName, inputName);
    net.addLayer(sprintf('my_outsig'), dagnn.Sigmoid(),...
        inputName, outputName);
    varCurT = varCurT+1;
elseif outputSoftmax
    inputName = sprintf('x%s%d',layerT,varCurT);
    outputName = 'prediction';
    net.renameVar(outputName, inputName);
    net.addLayer(sprintf('my_outsfx'), dagnn.SoftMax(),...
        inputName, outputName);
    varCurT = varCurT+1;
end

varSizes = net.getVarSizes({'data',opts.imageSize})';

deltaY = 1;
for i=0:1
    net.params(end-i).value=net.params(end-i).value/deltaY;
end


% net.addLayer('s', my_EstimationLoss('lossType', 'l2loss'),{'prediction','label'}, 's') ;
% net.addLayer('s', dagnn.Loss('loss', 'softmaxlog'),{'prediction','label'}, 's') ;
% net.addLayer('loss', dagnn.Loss('loss', 'softmaxlog'), {'prediction','label'}, 'loss') ;
% net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), {'prediction','label'}, 'top1err') ;
% net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', 'opts', {'topK',5}), ...
%                  {'prediction','label'}, 'top5err') ;
% net.vars(net.getVarIndex('prediction')).precious = 1 ;

net.addLayer('loss', dagnn.Loss('loss', 'hinge'), {'prediction','label'}, 'loss') ;
% net.addLayer('loss', dagnn.Loss('loss', 'logistic'), {'prediction','label'}, 'loss') ;
net.addLayer('binerror', dagnn.Loss('loss', 'binaryerror'), {'prediction','label'}, 'binerror') ;
net.vars(net.getVarIndex('prediction')).precious = 1 ;

% -------------------------------------------------------------------------
%                                                           Meta parameters
% -------------------------------------------------------------------------

net.meta.normalization.imageSize = opts.imageSize ;
net.meta.inputSize = [net.meta.normalization.imageSize, 32] ;
net.meta.normalization.cropSize = net.meta.normalization.imageSize(1) / 256 ;
net.meta.normalization.averageImage = opts.averageImage ;

net.meta.classes.name = opts.classNames ;
net.meta.classes.description = opts.classDescriptions ;

net.meta.augmentation.jitterLocation = true ;
net.meta.augmentation.jitterFlip = true ;
net.meta.augmentation.jitterBrightness = double(0.1 * opts.colorDeviation) ;
net.meta.augmentation.jitterAspect = [3/4, 4/3] ;
net.meta.augmentation.jitterScale  = [0.4, 1.1] ;
%net.meta.augmentation.jitterSaturation = 0.4 ;
%net.meta.augmentation.jitterContrast = 0.4 ;

net.meta.inputSize = {'data', [net.meta.normalization.imageSize 32]} ;

%lr = logspace(-1, -3, 60) ;
lr = [0.1 * ones(1,30), 0.01*ones(1,30), 0.001*ones(1,30)] ;
net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.momentum = 0.9 ;
net.meta.trainOpts.batchSize = 256 ;
net.meta.trainOpts.numSubBatches = 4 ;
net.meta.trainOpts.weightDecay = 0.0001 ;

% Init parameters randomly
% net.initParams() ;

for l = 1:numel(net.layers)
  if isa(net.layers(l).block, 'dagnn.BatchNorm')
    k = net.getParamIndex(net.layers(l).params{3}) ;
    net.params(k).learningRate = 0.3 ;
  end
end

end
