function imdb = my_setup_na(opts)

imdb.paths.image = fullfile(opts.dataDir, '*.mat') ;

imdb.sets.id = uint8([1 2 3]) ;
imdb.sets.name = {'train', 'val', 'test'} ;
imdb.classes.id = [];
imdb.classes.name = {};
imdb.classes.images = cell(1,20) ;
imdb.images.id = [] ;
imdb.images.name = {} ;
imdb.images.set = [] ;
imdb = addImageSet(imdb) ;
imdb = addData(imdb, opts) ;

% Compress data types
imdb.images.id = uint32(imdb.images.id) ;
imdb.images.set = uint8(imdb.images.set) ;

% -------------------------------------------------------------------------
function [imdb] = addData(imdb, opts)
% -------------------------------------------------------------------------
rng(opts.neuroN);
n = opts.sampleN*opts.neuroN;
% xyzis = [rand(n,3)*2-1,rand(n,1)*0.9+0.1];
tmp = randn(n,3);
xyzis = [mod(tmp,sign(tmp)),rand(n,1)*0.9+0.1];
imdb.xyzis = reshape(xyzis, [opts.sampleN,opts.neuroN,4]);

% -------------------------------------------------------------------------
function [imdb] = addImageSet(imdb)
% -------------------------------------------------------------------------
j = length(imdb.images.id) ;
names = dir(imdb.paths.image);
for i=1:length(names)
    j = j + 1 ;
    imdb.images.id(j) = j ;
    imdb.images.set(j) = 0 ;
    imdb.images.name{j} = names(i).name ;
    imdb.images.classification(j) = true ;
end

% -------------------------------------------------------------------------
function str=esc(str)
% -------------------------------------------------------------------------
str = strrep(str, '\', '\\') ;

% -------------------------------------------------------------------------
function str=cse(str)
% -------------------------------------------------------------------------
str = strrep(str, '\\', '\') ;
