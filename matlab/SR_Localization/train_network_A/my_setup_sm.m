function imdb = my_setup_sm( opts, bopts )

% imdb.paths.image = fullfile(opts.dataDir, '*.mat') ;

imdb.sets.id = uint8([1 2 3]) ;
imdb.sets.name = {'train', 'val', 'test'} ;
imdb.classes.id = [];
imdb.classes.name = {};
imdb.classes.images = cell(1,20) ;
imdb.images.id = [] ;
imdb.images.name = {} ;
imdb.images.set = [] ;

cal = opts.cal;
numlocs=opts.sampleN; %numlocs: number of simulated molecules.
rng(numlocs);
RoiPixelsize=bopts.imageSize(1); %ROI in pixels for simulation
dz=cal.cspline.dz;  %coordinate system of spline PSF is corner based and in units pixels / planes
z0=cal.cspline.z0; % distance and midpoint of stack in spline PSF, needed to translate into nm coordinates
dx=RoiPixelsize/2; %distance of center from edge
zRange = [-1 1]*500;
% ground_truth.z=linspace(zRange(1),zRange(2),numlocs)'; %define some coordinates. Alternatively, use rand
% ground_truth.x=rand(numlocs,1)*2-1;
% ground_truth.y=rand(numlocs,1)*2-1;
% coordinates=horzcat(ground_truth.x+dx,ground_truth.y+dx,ground_truth.z/dz+z0);
Intensity=bopts.Intensity; %number of photons / localization
background=bopts.background; %number of background photons per pixel

bgRange = bopts.bgRange;
density = bopts.density;
densities = density(randi(length(density),numlocs,1));
gt_bg.x = []; gt_bg.y = []; gt_bg.z = []; cor_bg = [];
for i=1:max(density)
    x_tmp = (rand(numlocs,1)*2-1)*bgRange*RoiPixelsize/2; 
    y_tmp = (rand(numlocs,1)*2-1)*bgRange*RoiPixelsize/2; 
    z_tmp = rand(numlocs,1)*(zRange(2)-zRange(1))+zRange(1); 
    
    if i==1
        fixMask = rand(numlocs,1)<bopts.fixXY;
        x_tmp(fixMask) = x_tmp(fixMask)/RoiPixelsize; 
        y_tmp(fixMask) = y_tmp(fixMask)/RoiPixelsize;
    end
    
    zeroMask = densities<i;
    x_tmp(zeroMask) = 0; y_tmp(zeroMask) = 0; z_tmp(zeroMask) = 0;
    
    gt_bg.x = [gt_bg.x x_tmp];
    gt_bg.y = [gt_bg.y y_tmp];
    gt_bg.z = [gt_bg.z z_tmp];
    
    cor_bg = horzcat(cor_bg,horzcat(x_tmp+dx,y_tmp+dx,z_tmp/dz+z0));
end

imstack = my_simSplinePSF(RoiPixelsize,cal.cspline.coeff,Intensity,background,cor_bg); %simulate images

% Compress data types
imdb.imstack = imstack;
imdb.ground_truth = [];
imdb.gt_bg = gt_bg;

end

