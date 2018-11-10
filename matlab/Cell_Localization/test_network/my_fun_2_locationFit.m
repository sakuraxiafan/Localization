function [ centers ] = my_fun_2_locationFit( preds_z_scale, thr_z, filterSize, maxSize, cMaxD )

if nargin<2 || isempty(thr_z), thr_z = -Inf; end;
if thr_z<1, thr_n = Inf; else, thr_n = thr_z; thr_z = -Inf; end;
if nargin<3 || isempty(filterSize), filterSize = 3; end;
if nargin<4 || isempty(maxSize), maxSize = [3 3 3]; end;
if nargin<5 || isempty(cMaxD), cMaxD = 30; end;

if filterSize==0, tmp = preds_z_scale;
else tmp = imgaussfilt3(preds_z_scale,filterSize); end;
% my_slice4D(tmp,thr_z,3);

% find local maximum
msk = true(maxSize);
msk(round(maxSize(1)/2),round(maxSize(2)/2),round(maxSize(3)/2)) = false;
tmp_dil = imdilate(tmp,msk);
M = (tmp >= tmp_dil) & (tmp > thr_z);
% [Mx, My, Mz]=ind2sub(size(M),find(permute(M,[2 1 3])));
[Mx, My, Mz]=ind2sub(size(M),find(M));
Mv = [Mx, My, Mz, tmp(M)];

% Mdmin = 30; 
overlap = [];
Mdist = bsxfun(@minus, permute(Mv(:,1:3),[2 1 3]), permute(Mv(:,1:3),[2 3 1]));
Mdist = squeeze(sum(Mdist.^2,1).^0.5);
for i=1:size(Mdist,1)
    list = find(Mdist(:,i)<cMaxD);
    [~, maxI] = max(Mv(list,4));
    list(maxI) = [];
    if ~isempty(list), overlap = cat(1, overlap, list); end;
end
centers = Mv; centers(unique(overlap),:) = [];
[~,I] = sort(centers(:,4),'descend');
centers = centers(I,:);
% centers = centers(:,1:3);
centers = centers(1:min(size(centers,1),thr_n),:);
centers = centers(:,1:ndims(preds_z_scale));

end

