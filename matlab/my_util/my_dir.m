function [ dirlist, dirnamelist, foldernamelist, filenamelist ] = my_dir( loaddir , fileType, sortFlag)
%MY_DIR Summary of this function goes here
%   Detailed explanation goes here

if nargin<2, fileType = []; end;
if nargin<3, sortFlag = false; end;

listTemp = dir([loaddir filesep fileType]);

Is = 1:length(listTemp);
if sortFlag
    listName = [];
    for i=1:length(listTemp)
        if isempty(fileType) && i<=2, tmp = {'-Inf'};
        else, tmp = regexp(listTemp(i).name, '\.', 'split'); tmp = regexp(tmp{1},'\d*\.?\d*','match');
        end
        listName = cat(2, listName, str2num(tmp{1})); 
    end
    if length(listName)==length(listTemp), [~,Is] = sort(listName); end;
end

dirlist = cell(0);
dirnamelist = cell(0);
foldernamelist = cell(0);
filenamelist = cell(0);
for i=Is
	if strcmp(listTemp(i).name, '.') || strcmp(listTemp(i).name, '..')
		continue;
	end
	
	dirnamelist{end+1} = listTemp(i).name;
	dirlist{end+1} = [loaddir filesep dirnamelist{end}];
	
	if listTemp(i).isdir
		foldernamelist{end+1} = listTemp(i).name;
	else
		filenamelist{end+1} = listTemp(i).name;
	end

end

end

