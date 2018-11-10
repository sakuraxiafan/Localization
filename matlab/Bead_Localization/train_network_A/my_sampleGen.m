function [ img, x ] = my_sampleGen( xyzi, h, normFlag, convM, r, areas, area, rect )
%UNTITLED �˴���ʾ�йش˺����ժ�?
%   �˴���ʾ��ϸ˵��

if nargin<3 || isempty(normFlag), normFlag = false; end;
if nargin<4 || isempty(convM), convM = 'same'; end;
if nargin<5 || isempty(r), r = 2; end;
if nargin<6 || isempty(areas), areas = [64 64 128]; end;
if nargin<7 || isempty(area), area = [64 64 128]; end;
if nargin<8 || isempty(rect), rect = [64 64 128]; end;

if ~ismatrix(xyzi), error('dimention of input is larger than 3.'); end;
if size(xyzi,2)==1, xyzi = xyzi'; end;


x = zeros([areas(1) areas(2) areas(3)]);

xyzi_T=[];
for i=1:3
    xyzi_T = cat(2, xyzi_T, ...
        min(max(r+1,round((area(i)-r)*xyzi(:,i)/2+size(x,i)/2)),size(x,i)-r));
end
xyzi_T = cat(2, xyzi_T, xyzi(:,4));

for i=1:size(xyzi_T,1)
    x(xyzi_T(i,1)+(-r:r),xyzi_T(i,2)+(-r:r),xyzi_T(i,3)+(-r:r)) = ...
        xyzi_T(i,4)*ones((2*r+1)*ones(1,3))/(2*r+1)^3;
end

I = find(abs(sum(sum(x,1),2))>=1e-10);
if numel(h)>=64^3 % &&  length(I)<=25
    ys = zeros(size(x,1),size(x,2));
    for i=1:length(I)
        ys(:,:,i) = conv2fft(x(:,:,I(i)),h(:,:,I(i)),convM);
    end
    y_mid = sum(ys,3);
%     y_mid = ys(size(h,1)/2+(1:size(x,1)),size(h,2)/2+(1:size(x,2)));
else
    y = convnfft(single(x),single(h), convM);
    y_mid = y(:,:,floor(size(y,3)/2));
end


rate = [0.001 0.999];
if normFlag==-1
    img = real(y_mid);
    img = my_cut(img,rate(1),rate(2));
elseif normFlag==0
    img = real(log(y_mid)); img = my_thr(img,-20,0.1);
    img = my_cut(img,rate(1),rate(2));
    img = max(0, (img+20)/20);
elseif normFlag==1
    img = real(y_mid); 
    img = my_cut(img,rate(1),rate(2));
    img = (img-min(img(:)))/(max(img(:))-min(img(:)));
elseif normFlag==2
    img = real(log(y_mid)); img = my_thr(img,-20,0.1);
    img = my_cut(img,rate(1),rate(2));
    img = (img-min(img(:)))/(max(img(:))-min(img(:)));
end
% my_imagesc(img);
img = my_rect(img, rect);

end

function img = my_rect(img, rect)
sizes = size(img);
region = max(1,floor(sizes-rect)/2);
region = [region region+rect-1];
img = img(region(1):region(3),region(2):region(4));
end

function y = my_cut(x,rate1,rate2)
y = x;
v = sort(x(:));
m = [v(max(1,round(rate1*length(v)))) v(round(rate2*length(v)))];
y(y<m(1))=m(1); y(y>m(2))=m(2);
end


function y = my_thr(x,thr,rate)
y = x;
if all(x>=thr), return; end;

tmp = x(x>=thr);
sorted = sort(tmp(:));
value = sorted(floor(rate*length(sorted))+1);
y(x<=value) = value;
end

