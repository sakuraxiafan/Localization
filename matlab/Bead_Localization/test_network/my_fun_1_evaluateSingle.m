function [ preds ] = my_fun_1_evaluateSingle( nets, img, opts, expNoI )

iSz = size(img);
wSz = opts.windowSize;
wSt = opts.windowStride;
outs = opts.outs;

preds = [];

batchN = 20; batch = zeros([wSz, opts.imageSize(3), batchN]);
rcN = floor((iSz-wSz)./wSt)+1; rcI = 1;
while(rcI<=prod(rcN))
    progress('Processing', rcI, prod(rcN));
    
    ri = floor((rcI-1)/rcN(2))+1;
    ci = mod(rcI-1,rcN(2))+1;
    patch = img(ri*wSt(1)-wSt(1)+(1:wSz(1)),ci*wSt(2)-wSt(2)+(1:wSz(2)));
    
    % normalize
    rate = [0.001 0.999];
    patch = my_cut(patch,rate(1),rate(2));
    patch = (patch-min(patch(:)))/(max(patch(:))-min(patch(:)));
    
    batchI = mod(rcI-1,batchN)+1;
    batch(:,:,:,batchI) = repmat(patch, [1 1 opts.imageSize(3)]);
    if batchI==batchN || rcI==prod(rcN)
        im = single(batch(:,:,:,1:batchI));
        if opts.useGpu
            im = gpuArray(im);
        end
        inputs = {'data', im};
        
        nets{expNoI}.eval(inputs);
        pred_tmp = squeeze(gather(nets{expNoI}.vars(outs(expNoI)).value));
        if expNoI==2, preds = cat(1, preds, pred_tmp');
        else, preds = cat(1, preds, pred_tmp);
        end
            
    end
    rcI = rcI+1;
end

% save (ks) results
if expNoI==2
    preds = permute(reshape(preds, [fliplr(rcN) size(preds,2)]), [2 1 3]);
else
    preds = permute(reshape(preds, [fliplr(rcN) size(preds,2)]), [2 1]);
end


end

