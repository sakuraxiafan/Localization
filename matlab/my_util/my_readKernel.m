function [ kernel ] = my_readKernel( kernelFile, mode )

% kernelName = 'PSF BW_z250.tif';
% kernelName = 'PSF BW_z50.tif';
% kernelName = 'PSF BW_z20.tif';
% kernelName = 'PSF BW_xy100z100_hw256d65.tif';
% kernelFile = ['..' filesep 'simulated_psf' filesep kernelName];

if nargin<2, mode='double'; end;
showKelFlag = false;

% imread_multi
info = imfinfo(kernelFile);
kernel = [];
for i=1:length(info)
    tmp = imread(kernelFile, i);
    if strcmp(mode, 'double'), tmp=im2double(tmp); end;
    kernel = cat(3, kernel, tmp);
end

if showKelFlag
    figure,
    for i=1:size(kernel,3)
        imagesc(kernel(:,:,i)); title(sprintf('image\t%4d/%4d', i, size(kernel,3)));
        progress('Showing', i, size(kernel,3));
        pause(0.05);
    end
end

% maxs = squeeze(max(max(kernel)))';
% shows = 10; % 4;
% disp(maxs(round(end/2-shows:end/2+shows)));

end

