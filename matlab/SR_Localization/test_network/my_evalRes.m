function [ ress, pairs ] = my_evalRes( ress, xyzi, pred, MatchD, allFlag, matchDepths)

if nargin<5 || isempty(allFlag), allFlag = false; end
if nargin<6, matchDepths = []; end

if isempty(ress) || iscell(ress)
    res = struct('tp',0,'fp',0,'fn',0,'mae',0,'rmse',0,'pr',0,'rc',0,'f1',0,'mdae',0,'jac',0);
else
    res = ress;
end

if nargin<4 || isempty(MatchD), MatchD = Inf; end;

scaleXY = 1; scaleZ = 1;

scaleBefore = 0;

if scaleBefore
xyzi = [xyzi(:,1:2)*scaleXY xyzi(:,3)*scaleZ];
pred = [pred(:,1:2)*scaleXY pred(:,3)*scaleZ];
end

rate = 1;
if ~isempty(matchDepths)
    rate = size(xyzi,1);
    xyzi((xyzi(:,end)<matchDepths(1)) | (xyzi(:,end)>matchDepths(2)),:) = [];
    rate = size(xyzi,1)/rate;
end

dists = bsxfun(@minus,reshape(xyzi(:,1:3),[size(xyzi,1) 1 3]),...
    reshape(pred(:,1:3),[1,size(pred,1),3]));
dists = sum(dists.^2,3).^0.5;
pairs = [];
for i=1:min(size(dists))
    [minD,idx] = min(dists(:));
    if minD>MatchD, break; end
    [xi,yi] = ind2sub(size(dists),idx);
    pairs = cat(1,pairs,[xi,yi]);
    dists(xi,:)=Inf; dists(:,yi)=Inf;
end
tmp = [];
for i=1:size(pairs,1)
    locs = [xyzi(pairs(i,1),1:3); pred(pairs(i,2),1:3)];
    ae = abs([locs(1,:)-locs(2,:)]);
    if ~scaleBefore, ae = [ae(:,1:end-1)*scaleXY ae(:,end)*scaleZ]; end
    se = ae.^2;
    tmp = [tmp; ae];
    
    res.mae = (res.mae*res.tp+ae)/(res.tp+1);
    res.rmse = ((res.rmse.^2*res.tp+se)/(res.tp+1)).^.5;
    res.tp = res.tp+1;
end
if allFlag
    ii=res.tp;
    for i=1:size(xyzi,1)-size(pairs,1)
        ae = ones(1,3)*MatchD/3.^.5;
        if ~scaleBefore, ae = [ae(:,1:end-1)*scaleXY ae(:,end)*scaleZ]; end
        se = ae.^2;
        tmp = [tmp; ae];
        
        res.mae = (res.mae*ii+ae)/(ii+1);
        res.rmse = ((res.rmse.^2*ii+se)/(ii+1)).^.5;
        ii = ii+1;
    end
end
if isempty(tmp), res.mdae = 0; else, res.mdae = median(tmp,1); end
res.fp = res.fp+round(size(pred,1)*rate)-size(pairs,1);
res.fn = res.fn+size(xyzi,1)-size(pairs,1);

res.pr = res.tp/(res.tp+res.fp);
res.rc = res.tp/(res.tp+res.fn);
res.f1 = 2/(1/(res.pr+1e-7)+1/(res.rc+1e-7));
res.jac = res.tp/(res.tp+res.fp+res.fn);

if iscell(ress)
    ress{end+1} = res;
else
    ress = res;
end

end

