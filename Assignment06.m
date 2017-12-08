%% CEC495A
% Tymothy Anderson
% Assignment 06 - Artificial Neural Networks

clear all; clc; close all;

%% Training Image
% Clean up training image
Igray = imread('training.jpg');
Imed = medfilt2(Igray,[100,100]);
Ifinal = Imed - Igray;
BW1 = Ifinal > 5;
imshow(BW1);

% Label objects
[labels,number] = bwlabel(BW1,8);
Istats = regionprops(labels,'basic','Centroid');

% Remove dots
Istats( [Istats.Area] < 500 ) = [];
num = length(Istats);

% Make things better
Ibox = floor( [Istats.BoundingBox] );
Ibox = reshape(Ibox,[4 num]);

% Make baby images
for k = 1:num
    col1 = Ibox(1,k);
    col2 = Ibox(1,k) + Ibox(3,k);
    row1 = Ibox(2,k);
    row2 = Ibox(2,k) + Ibox(4,k);
    subImage = BW1(row1:row2, col1:col2);
    subImageScaled = imresize(subImage, [24,12]);
    TPattern(k,:) = subImageScaled(:)';
end

% Generate target matrix
TTarget = zeros(100,10);
for row = 1:10
    for col = 1:10
        TTarget( 10*(row-1) + col, row ) = 1;
    end
end

% Make matrices readable by neural network
TPattern = TPattern';
TTarget = TTarget';

% Train the neural network
net = newff(...
    [zeros(288,1) ones(288,1)],...
    [24 10],...
    {'logsig' 'logsig'},...
    'traingdx' );
net.trainParam.epochs  = 500;
net = train(net,TPattern,TTarget);
view(net);

%% Unknown Image
% Clean up unknown image
for num = [603032, 196128, 480000, 480096,]
    Igray = imread(sprintf('%d.jpg',num));
    Imed = medfilt2(Igray,[100,100]);
    Ifinal = Imed - Igray;
    BW1 = Ifinal > 5;
    imshow(BW1);
    
    % Label objects
    [labels,number] = bwlabel(BW1,8);
    Istats = regionprops(labels,'basic','Centroid');
    
    % Remove dots
    Istats( [Istats.Area] < 1000 ) = [];
    num = length(Istats);
    
    % Make things better
    Ibox = floor( [Istats.BoundingBox] );
    Ibox = reshape(Ibox,[4 num]);
    
    % Make baby images
    for k = 1:num
        col1 = Ibox(1,k);
        col2 = Ibox(1,k) + Ibox(3,k);
        row1 = Ibox(2,k);
        row2 = Ibox(2,k) + Ibox(4,k);
        subImage = BW1(row1:row2, col1:col2);
        subImageScaled = imresize(subImage, [24,12]);
        UPattern(k,:) = subImageScaled(:)';
    end
    
    % Generate target matrix
    UTarget = zeros(100,10);
    for row = 1:10
        for col = 1:10
            UTarget( 10*(row-1) + col, row ) = 1;
        end
    end
    
    % Make matrices readable by neural network
    UPattern = UPattern';
    UTarget = UTarget';
    
    % Test network with unknown image
    Y = sim(net,UPattern);
    [value, index] = max( Y(:,:) );
    index-1  % Print detected vector
    clear UPattern
end