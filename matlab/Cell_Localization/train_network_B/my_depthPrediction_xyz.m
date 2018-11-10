function [imgs, preds, gts, preds_xy, gts_xy] = ...
    my_depthPrediction_xyz(net, imdb, getBatch, val, opts, varargin)

% Depth prediction (inference) using a trained model.
% Inputs (imdb) can be either from the NYUDepth_v2 or Make3D dataset, along
% with the corresponding trained model (net). Additionally, the evaluation
% can be run for any single image. MatConvNet library has to be already
% setup for this function to work properly.
% -------------------------------------------------------------------------
% Inputs:
%   - imdb: a structure with fields 'images' and 'depths' in the case of
%           the benchmark datasets with known ground truth. imdb could
%           alternatively be any single RGB image of size NxMx3 in [0,255]
%           or a tensor of D input images NxMx3xD.
%   - net:  a trained model of type struct (suitable to be converted to a 
%           DagNN object and successively processed using the DagNN 
%           wrapper). For testing on arbitrary images, use NYU model for 
%           indoor and Make3D model for outdoor scenes respectively.
% -------------------------------------------------------------------------

netOpts.gpu = false;
netOpts.plot = false;          % Set to true to visualize the predictions during inference
netOpts.mode = 'test';
netOpts = vl_argparse(netOpts, varargin);

% Set network properties
net = dagnn.DagNN.loadobj(net);
% net.mode = 'test';
% net.mode = 'normal';
net.mode = netOpts.mode;
out = net.getVarIndex('prediction');
out_xy = net.getVarIndex('prediction_xy');
if netOpts.gpu
    net.move('gpu');
end

imgs = [];
preds = [];    % initiliaze
gts = [];    % initiliaze 
preds_xy = [];    % initiliaze
gts_xy = [];    % initiliaze 

% fprintf('predicting...\n');
for i = 1:length(val)
    
    y = getBatch(imdb, val(i)) ;
    images = y{2};
    gt_tmp = y{4};
    gt_xy_tmp = y{6};
    imgs = cat(4, imgs, gather(images));
    gts = cat(4, gts, gt_tmp);
    gts_xy = cat(4, gts_xy, gt_xy_tmp);
    
    % get input image
    im = single(images);
    if netOpts.gpu
        im = gpuArray(im);
    end
    
    % run the CNN
%     inputs = {'data', im};
    inputs = {'data', im, 'label', gt_tmp, 'label_xy', gt_xy_tmp};
    net.eval(inputs) ;
    
    % obtain prediction
    pred_tmp = gather(net.vars(out).value);
    preds = cat(4, preds, pred_tmp);
%     pred_xy_tmp = gather(net.vars(out_xy).value);
%     preds_xy = cat(4, preds_xy, pred_xy_tmp);
    
    progress('predicting',i,length(val));
    
    
end

imgs = bsxfun(@plus,imgs,net.meta.normalization.rgbMean);

end