function [imgs, preds, gts ] = my_prediction_xyz( net, imdb, getBatch, val, opts, varargin )

netOpts.gpu = false;
netOpts.plot = false;          % Set to true to visualize the predictions during inference
netOpts.mode = 'test';
netOpts.testBatch = 64;
netOpts = vl_argparse(netOpts, varargin);

% Set network properties
net = dagnn.DagNN.loadobj(net);
net.mode = netOpts.mode;
out = net.getVarIndex('prediction');
if netOpts.gpu
    net.move('gpu');
end

imgs = [];
preds = [];    % initiliaze
gts = [];    % initiliaze 

is = 1:ceil(length(val)/netOpts.testBatch);
for i = 1:length(is)
    idx = (i-1)*netOpts.testBatch+(1:netOpts.testBatch);
    idx(idx>length(val)) = [];
    
    y = getBatch(imdb, val(idx)) ;
    images = y{2};
    gt_tmp = y{4};
    imgs = cat(4, imgs, gather(images));
    gts = cat(4, gts, gt_tmp);
    
    % get input image
    im = single(images);
    if netOpts.gpu
        im = gpuArray(im);
    end
    
    % run the CNN
    inputs = {'data', im, 'label', gt_tmp};
    net.eval(inputs) ;
    
    % obtain prediction
    pred_tmp = gather(net.vars(out).value);
    preds = cat(4, preds, pred_tmp);
    
    progress('predicting',i,length(is));
end

end

