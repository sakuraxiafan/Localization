function predTs = my_kalmanTracking(centersList_tmp,paraT)

% paraT.costOfNonAssignment = 20;
% paraT.invisibleForTooLong = 20;
% paraT.ageThreshold = 8;
% paraT.iniError = [200,50];
% paraT.MotNoise = [100,25];
% paraT.MeaNoise = 100;
        
predTsAll = [];

tracks = initializeTracks(); % Create an empty array of tracks.
nextId = 1; % ID of the next track
% Detect moving objects, and track them across video frames.
for i = min(centersList_tmp(:,end)):max(centersList_tmp(:,end))
    
    centroids = centersList_tmp(centersList_tmp(:,end)==i,1:end-1);
    tracks = predictNewLocationsOfTracks(tracks);
    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment(tracks,centroids,paraT);
    
    tracks = updateAssignedTracks(tracks,assignments,centroids);
    tracks = updateUnassignedTracks(tracks,unassignedTracks);
    tracks = deleteLostTracks(tracks,paraT);
    [tracks,nextId] = createNewTracks(tracks,centroids,unassignedDetections,nextId,paraT);
    
    for j=1:length(tracks)
        if ~tracks(j).consecutiveInvisibleCount
            predTsAll = cat(1,predTsAll,[tracks(j).bbox,i,tracks(j).id]);
        end
    end
end

predTsAll = sortrows(predTsAll,5);
predTs = []; ii = 1;
for i=unique(predTsAll(:,end))'
    frames = predTsAll(predTsAll(:,end)==i,end-1);
    if frames(end)-frames(1)>=paraT.ageThreshold
        predTs = cat(1,predTs,...
            [predTsAll(predTsAll(:,end)==i,1:end-1) ones(size(frames))*ii]);
        ii = ii+1;
    end
end
end

function tracks = initializeTracks()
% create an empty array of tracks
tracks = struct(...
    'id', {}, ...
    'bbox', {}, ...
    'kalmanFilter', {}, ...
    'age', {}, ...
    'totalVisibleCount', {}, ...
    'consecutiveInvisibleCount', {});
end

function tracks = predictNewLocationsOfTracks(tracks)
for i = 1:length(tracks)
    bbox = tracks(i).bbox;
    
    % Predict the current location of the track.
    predictedCentroid = predict(tracks(i).kalmanFilter);
    
    % Shift the bounding box so that its center is at
    % the predicted location.
    %             predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
    %             tracks(i).bbox = [predictedCentroid, bbox(3:4)];
    tracks(i).bbox = predictedCentroid;
end
end

function [assignments, unassignedTracks, unassignedDetections] = ...
    detectionToTrackAssignment(tracks,centroids,paraT)

nTracks = length(tracks);
nDetections = size(centroids, 1);

% Compute the cost of assigning each detection to each track.
cost = zeros(nTracks, nDetections);
for i = 1:nTracks
    cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
end

% Solve the assignment problem.
costOfNonAssignment = paraT.costOfNonAssignment;
[assignments, unassignedTracks, unassignedDetections] = ...
    assignDetectionsToTracks(cost, costOfNonAssignment);
end


    function tracks = updateAssignedTracks(tracks,assignments,centroids)
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            centroid = centroids(detectionIdx, :);
            bbox = centroids(detectionIdx, :);
            
            % Correct the estimate of the object's location
            % using the new detection.
            correct(tracks(trackIdx).kalmanFilter, centroid);
            
            % Replace predicted bounding box with detected
            % bounding box.
            tracks(trackIdx).bbox = bbox;
            
            % Update track's age.
            tracks(trackIdx).age = tracks(trackIdx).age + 1;
            
            % Update visibility.
            tracks(trackIdx).totalVisibleCount = ...
                tracks(trackIdx).totalVisibleCount + 1;
            tracks(trackIdx).consecutiveInvisibleCount = 0;
        end
    end

    function tracks = updateUnassignedTracks(tracks,unassignedTracks)
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            tracks(ind).age = tracks(ind).age + 1;
            tracks(ind).consecutiveInvisibleCount = ...
                tracks(ind).consecutiveInvisibleCount + 1;
        end
    end

    function tracks = deleteLostTracks(tracks,paraT)
        if isempty(tracks)
            return;
        end
        
        invisibleForTooLong = paraT.invisibleForTooLong;
        ageThreshold = paraT.ageThreshold;
        
        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;
        
        % Find the indices of 'lost' tracks.
        lostInds = (ages < ageThreshold & visibility < 0.6) | ...
            [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;
        
        % Delete lost tracks.
        tracks = tracks(~lostInds);
    end

    function [tracks,nextId] = createNewTracks(tracks,centroids,unassignedDetections,nextId,paraT)
        bboxes = centroids(unassignedDetections, :);
        centroids = centroids(unassignedDetections, :);
        iniError = paraT.iniError;
        MotNoise = paraT.MotNoise;
        MeaNoise = paraT.MeaNoise;
        for i = 1:size(centroids, 1)
            
            centroid = centroids(i,:);
            bbox = bboxes(i, :);
            
            % Create a Kalman filter object.
            %  kalmanFilter = configureKalmanFilter(MotionModel, ...
            %  InitialLocation, InitialEstimateError, MotionNoise, MeasurementNoise)
            kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                centroid, iniError, MotNoise, MeaNoise);
            
            % Create a new track.
            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);
            
            % Add it to the array of tracks.
            tracks(end + 1) = newTrack;
            
            % Increment the next id.
            nextId = nextId + 1;
        end
    end
