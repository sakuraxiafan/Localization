function [ net ] = my_cnnInitializeModel_xyz( varargin )

opts.imageSize = [63, 63, 3];
opts.outputSize = [1, 1, 100];
opts.outputDepth = opts.outputSize(3);
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.model = '';

opts.classNames = {} ;
opts.classDescriptions = {} ;
opts.averageImage = zeros(opts.imageSize(3),1) ;
opts.colorDeviation = zeros(opts.imageSize(3)) ;
opts.cudnnWorkspaceLimit =  640 * 1024^3 ; %1024*1024*1024 ;
[opts, ~] = vl_argparse(opts, varargin) ;


opts.scale = 1 ;
opts.initBias = 0 ;
opts.weightDecay = 1 ;
opts.weightInitMethod = 'gaussian' ;
opts.batchNormalization = false ;
opts.networkType = 'simplenn' ;
[opts, ~]  = vl_argparse(opts, varargin) ;

net.meta.normalization.imageSize = opts.imageSize;
net = my_net(net, opts);
net = dagnn.DagNN.fromSimpleNN(net);
net.renameVar('x0','data');
net.renameVar(net.vars(end).name, 'prediction');
net.vars(net.getVarIndex('prediction')).precious = 1;

net.addLayer('loss', dagnn.Loss('loss', 'hinge'), {'prediction','label'}, 'loss') ;
net.addLayer('binerror', dagnn.Loss('loss', 'binaryerror'), {'prediction','label'}, 'binerror') ;


deltaY = 2.5;
% for i=0:1
for i=0:length(net.params)-1
    net.params(end-i).value=net.params(end-i).value*deltaY;
end

% --------------------------------------------------------------------
function net = add_block(net, opts, id, h, w, in, out, stride, pad)
% --------------------------------------------------------------------
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
  name = 'fc' ;
else
  name = 'conv' ;
end
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
                           'weights', {{init_weight(opts, h, w, in, out, 'single'), ...
                             ones(out, 1, 'single')*opts.initBias}}, ...
                           'stride', stride, ...
                           'pad', pad, ...
                           'dilate', 1, ...
                           'learningRate', [1 2], ...
                           'weightDecay', [opts.weightDecay 0], ...
                           'opts', {convOpts}) ;
if opts.batchNormalization
  net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%s',id), ...
                             'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single'), ...
                               zeros(out, 2, 'single')}}, ...
                             'epsilon', 1e-4, ...
                             'learningRate', [2 1 0.1], ...
                             'weightDecay', [0 0]) ;
end
net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;

% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

switch lower(opts.weightInitMethod)
  case 'gaussian'
    sc = 0.01/opts.scale ;
    weights = randn(h, w, in, out, type)*sc;
  case 'xavier'
    sc = sqrt(3/(h*w*in)) ;
    weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
  case 'xavierimproved'
    sc = sqrt(2/(h*w*out)) ;
    weights = randn(h, w, in, out, type)*sc ;
  otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end

% --------------------------------------------------------------------
function net = add_pool(net, opts, id, h, w, stride, pad)
% --------------------------------------------------------------------
net.layers{end+1} = struct('type', 'pool', 'name', sprintf('pool%s',id), ...
                           'method', 'max', ...
                           'pool', [h w], ...
                           'stride', stride, ...
                           'pad', pad) ;

% --------------------------------------------------------------------
function net = add_dropout(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'dropout', ...
                             'name', sprintf('dropout%s', id), ...
                             'rate', 0.5) ;
end


% --------------------------------------------------------------------
function net = my_net(net, opts)
% --------------------------------------------------------------------

net.layers = {} ;

net = add_block(net, opts, '1', 3, 3, 1, 64, 1, 1) ;
net = add_block(net, opts, '2', 3, 3, 64, 256, 1, 1) ;
net = add_pool(net, opts, '2', 2, 2, 2, 0) ;

net = add_block(net, opts, '3', 3, 3, 256, 512, 1, 1) ;
net = add_block(net, opts, '4', 3, 3, 512, 512, 1, 1) ;
net = add_pool(net, opts, '4', 2, 2, 2, 0) ;

net = add_block(net, opts, '5', 3, 3, 512, 1024, 1, 0) ;
net.layers(end) = [] ; net = add_dropout(net, opts, '5') ;

net = add_block(net, opts, '6', 1, 1, 1024, opts.outputSize(3) , 1, 0) ;
net.layers(end) = [] ;

