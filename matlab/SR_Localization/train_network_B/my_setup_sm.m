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
dx=floor(RoiPixelsize/2); %distance of center from edge
zRange = [-1 1]*500;
ground_truth.z=linspace(zRange(1),zRange(2),numlocs)'; %define some coordinates. Alternatively, use rand
% ground_truth.x=linspace(-0.5,0.5,numlocs)';
% ground_truth.y=sin(ground_truth.x*4*pi);
ground_truth.x=rand(numlocs,1)*2-1;
ground_truth.y=rand(numlocs,1)*2-1;
coordinates=horzcat(ground_truth.x+dx,ground_truth.y+dx,ground_truth.z/dz+z0);
Intensity=bopts.Intensity; %number of photons / localization
background=bopts.background; %number of background photons per pixel

bgRange = bopts.bgRange;
density = bopts.density;
gt_bg.x = []; gt_bg.y = []; gt_bg.z = []; cor_bg = [];
for i=1:(density-1)
    x_tmp = (rand(numlocs,1)*2-1)*bgRange*RoiPixelsize/2; gt_bg.x = [gt_bg.x x_tmp];
    y_tmp = (rand(numlocs,1)*2-1)*bgRange*RoiPixelsize/2; gt_bg.y = [gt_bg.y y_tmp];
    z_tmp = rand(numlocs,1)*(zRange(2)-zRange(1))+zRange(1); gt_bg.z = [gt_bg.z z_tmp];
    cor_bg = horzcat(cor_bg,horzcat(x_tmp+dx,y_tmp+dx,z_tmp/dz+z0));
end

[imstack, zeroIdx] = my_simSplinePSF(RoiPixelsize,cal.cspline.coeff,Intensity,background,horzcat(coordinates,cor_bg)); %simulate images

for i=1:size(zeroIdx,1)
    gt_bg.x(zeroIdx(i,1),zeroIdx(i,2))=0;
    gt_bg.y(zeroIdx(i,1),zeroIdx(i,2))=0;
    gt_bg.z(zeroIdx(i,1),zeroIdx(i,2))=0;
end

% Compress data types
imdb.imstack = imstack;
imdb.ground_truth = ground_truth;
imdb.gt_bg = gt_bg;

end

