%  Copyright (c)2017 Ries Lab, European Molecular Biology Laboratory,
%  Heidelberg.
%  
%  This file is part of GPUmleFit_LM Fitter.
%  
%  GPUmleFit_LM Fitter is free software: you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation, either version 3 of the License, or
%  (at your option) any later version.
%  
%  GPUmleFit_LM Fitter is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%  
%  You should have received a copy of the GNU General Public License
%  along with GPUmleFit_LM Fitter.  If not, see <http://www.gnu.org/licenses/>.
%  
%  
%  Additional permission under GNU GPL version 3 section 7

%%
function [out, zeroIdx, masks] = my_simSplinePSF(Npixels,coeff,I,bg,cor,overlapFlag)
t=tic;
if (nargin <5)
   error('Minimal usage: simSplinePSF(Npixels,coeff,I,bg,cor)');
end

if nargin<6 || isempty(overlapFlag), overlapFlag = false; end

zeroIdx = [];

Nfits = size(cor,1);
spline_xsize = size(coeff,1);
spline_ysize = size(coeff,2);
spline_zsize = size(coeff,3);
off = floor(((spline_xsize+1)-Npixels)/2);
data = zeros(Npixels,Npixels,Nfits,'single')+bg;

density = size(cor,2)/3;
masks = zeros(Npixels,Npixels,density,Nfits,'logical');

for kk = 1:Nfits
    if overlapFlag, progress('Generating', kk, Nfits); end
    for di = 1:density
        
    xcenter = cor(kk,1+(di-1)*3);
    ycenter = cor(kk,2+(di-1)*3);
    zcenter = cor(kk,3+(di-1)*3);
    
    if xcenter==Npixels/2 && ycenter==Npixels/2, continue; end
    
    xc = -1*(xcenter - Npixels/2+0.5);
    yc = -1*(ycenter - Npixels/2+0.5);
    zc = zcenter - floor(zcenter);
    
%     if di>=2 && abs(xc)<=1 && abs(yc)<=1, continue; end
    if di>=2 && ((cor(kk,1)-xcenter)^2+(cor(kk,2)-ycenter)^2)^.5<=3
        zeroIdx(end+1,:) = [kk, di-1];
        if ~overlapFlag, continue; end
    end
    
    xstart = floor(xc);
    xc = xc - xstart;
    
    ystart = floor(yc);
    yc = yc - ystart;
    

    zstart = floor(zcenter);
    
   
    [delta_f,delta_dxf,delta_ddxf,delta_dyf,delta_ddyf,delta_dzf,delta_ddzf]=computeDelta3Dj_v2(single(xc),single(yc),single(zc));
    
    for ii = 0:Npixels-1
        for jj = 0:Npixels-1
             temp = fAt3Dj_v2(ii+xstart+off,jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,coeff);
             model = temp*I;
             data(ii+1,jj+1,kk)=data(ii+1,jj+1,kk)+model;
             if temp>5e-3, masks(ii+1, jj+1, di, kk) = 1; end
        end
    end
    if toc(t)>1
%         disp(kk/Nfits)
        t=tic;
    end
    end
end
out = (poissrnd(data,Npixels,Npixels,Nfits)); 
% out = data; 

