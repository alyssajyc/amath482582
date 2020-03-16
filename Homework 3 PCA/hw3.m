% N is the test number 
% Test 1: Ideal signal
% Test 2: Noisy signal
% Test 3: Horizontal displacement
% Test 4: Horizontal displacement and rotation
N = 1;    

% Load three video data of the same test
load(['cam1_' num2str(N) '.mat']);
load(['cam2_' num2str(N) '.mat'])
load(['cam3_' num2str(N) '.mat'])

% Use the same name for different tests
switch N
    case 1
        vidFrames1 = vidFrames1_1;
        vidFrames2 = vidFrames2_1;
        vidFrames3 = vidFrames3_1;
    case 2
        vidFrames1 = vidFrames1_2;
        vidFrames2 = vidFrames2_2;
        vidFrames3 = vidFrames3_2;
    case 3
        vidFrames1 = vidFrames1_3;
        vidFrames2 = vidFrames2_3;
        vidFrames3 = vidFrames3_3;
    otherwise
        vidFrames1 = vidFrames1_4;
        vidFrames2 = vidFrames2_4;
        vidFrames3 = vidFrames3_4;
end

% Find and click the first position of the flashlight

% Use the fourth element of size to find the lenghth of the video, and
% create a new vector
position1 = zeros(2,size(vidFrames1,4)); 
figure();
imshow(vidFrames1(:,:,:,1));
title('Click the whitest part of flashlight');
% Use ginput to click one point
[x1, y1] = ginput(1);
position1(:,1) = [y1; x1];

position2 = zeros(2,size(vidFrames2,4));
figure();
imshow(vidFrames2(:,:,:,1));
title('Click the whitest part of flashlight');
[x2, y2] = ginput(1);
position2(:,1) = [y2; x2];

position3 = zeros(2,size(vidFrames3,4));
figure()
imshow(vidFrames3(:,:,:,1));
title('Click the whitest part of flashlight');
[x3, y3] = ginput(1);
position3(:,1) = [y3; x3];

close all

% Set length and width of the rectangle that may contains the movement of 
% flashlight throughout the process 

% Camera 1
% Set the length
lengthsize1 = 20;
%Set the width
widthsize1 = 20;                         

% Camera 2
lengthsize2 = 20;                         
widthsize2 = 20;                          

% Camera 3
lengthsize3 = 20;                             
widthsize3 = 20;                          

% Set the size that may extend the video
lengthsizeextend = 0;
widthsizeextend = 0;

% Affirm rectangles that contain the movement of the whitest part of 
% flashlight 
    
% Camera 1
for l=2:size(vidFrames1,4)
    % Previous length position of the whitest part of flashlight
    length1 = position1(1,l-1);            
    % Previous width position of the whitest part of flashlight
    width1 = position1(2,l-1);             
    
    % Adjust the rectangle, so it won't extend the video
    if (length1 - lengthsize1) < 1
        lengthsizeextend = length1-1 + (length1==1);
    elseif (length1 + lengthsize1) > size(vidFrames1,1)
        lengthsizeextend = size(vidFrames1,1) - length1 + (length1 == size(vidFrames1,1));
    else
        lengthsizeextend = lengthsize1;
    end
    if (width1 - widthsize1) < 1
        widthsizeextend = width1-1 + (width1==1);
    elseif (width1 + widthsize1) > size(vidFrames1,1)
        widthsizeextend = size(vidFrames1,1) - width1 + (width1 == size(vidFrames1,1));
    else
        widthsizeextend = widthsize1;
    end
        
    
    % Find the location of previous bucket in the rectangle 
    local1 = vidFrames1((length1-lengthsizeextend):(length1+lengthsizeextend),(width1-widthsizeextend):(width1+widthsizeextend),:,l);
    
    % Find the maximum of all the location
    local_filtered1 = (sum(local1,3)==max(max(sum(local1,3))));
    
    % Use the maximum of all the location 
    [length,width] = find(local_filtered1);
    position1(1,l) = length(1) + (length1-lengthsizeextend)-1;
    position1(2,l) = width(1) + (width1-widthsizeextend)-1;
end


% camera 2
for l=2:size(vidFrames2,4)
    % Previous length position of the whitest part of flashlight
    length2 = position2(1,l-1); 
    % Previous width position of the whitest part of flashlight
    width2 = position2(2,l-1);             
    
    % Adjust the rectangle, so it won't extend the video
    if (length2 - lengthsize2) < 1
        lengthsizeextend = length2-1 + (length2==1);
    elseif (length2 + lengthsize2) > size(vidFrames2,1)
        lengthsizeextend = size(vidFrames2,1) - length2 + (length2 == size(vidFrames2,1));
    else
        lengthsizeextend = lengthsize2;
    end
    if (width2 - widthsize2) < 1
        widthsizeextend = width2-1 + (width2==1);
    elseif (width2 + widthsize2) > size(vidFrames2,1)
        widthsizeextend = size(vidFrames2,1) - width2 + (width2 == size(vidFrames2,1));
    else
        widthsizeextend = widthsize2;
    end
    
    % Find the location of previous bucket in the rectangle 
    local2 = vidFrames2((length2-lengthsizeextend):(length2+lengthsizeextend),(width2-widthsizeextend):(width2+widthsizeextend),:,l);
    
    % Find the maximum of all the location
    local_filtered2 = (sum(local2,3)==max(max(sum(local2,3))));
    
     % Use the maximum of all the location 
    [length,width] = find(local_filtered2);
    position2(1,l) = length(1) + (length2-lengthsizeextend)-1;
    position2(2,l) = width(1) + (width2-widthsizeextend)-1;
end


% camera 3
for l=2:size(vidFrames3,4)
    % Previous length position of the whitest part of flashlight
    lenght3 = position3(1,l-1);  
    % Previous width position of the whitest part of flashlight
    width3 = position3(2,l-1);             
    
    % Adjust the rectangle, so it won't extend the video
    if (lenght3 - lengthsize3) < 1
        lengthsizeextend = lenght3-1 + (lenght3==1);
    elseif (lenght3 + lengthsize3) > size(vidFrames3,1)
        lengthsizeextend = size(vidFrames3,1) - lenght3 + (lenght3 == size(vidFrames3,1));
    else
        lengthsizeextend = lengthsize3;
    end
    if (width3 - widthsize3) < 1
        widthsizeextend = width3-1 + (width3==1);
    elseif (width3 + widthsize3) > size(vidFrames3,1)
        widthsizeextend = size(vidFrames3,1) - width3 + (width3 == size(vidFrames3,1));
    else
        widthsizeextend = widthsize3;
    end
    
    % Find the location of previous bucket in the rectangle 
    local3 = vidFrames3((lenght3-lengthsizeextend):(lenght3+lengthsizeextend),(width3-widthsizeextend):(width3+widthsizeextend),:,l);
    
    % Find the maximum of all the location
    local_filtered3 = (sum(local3,3)==max(max(sum(local3,3))));
    
     % Use the maximum of all the location 
    [length,width] = find(local_filtered3);
    position3(1,l) = length(1) + (lenght3-lengthsizeextend)-1;
    position3(2,l) = width(1) + (width3-widthsizeextend)-1;
end

% Save data in .mat
save(['test' num2str(N) '.mat'], 'position1', 'position2', 'position3');