clc;
clear;
close all;

% N is the test number 
% Test 1: Ideal signal
% Test 2: Noisy signal
% Test 3: Horizontal displacement
% Test 4: Horizontal displacement and rotation
N = 3;

% Load the data by the first script
load(['test' num2str(N) '.mat'])

% Cut some of the data at the beginning to make the length of data in one
% test is the same
switch N
    case 1 
        position1 = position1(:,9:end);
        position2 = position2(:,18:end);
        position3 = position3(:,8:end);
    case 2
        position1 = position1(:,12:end);
        position2 = position2(:,:);
        position3 = position3(:,12:end);
    case 3
        position1 = position1(:,12:end);
        position2 = position2(:,1:end);
        position3 = position3(:,4:end);
    otherwise
        position1 = position1(:,2:end);
        position2 = position2(:,10:end);
        position3 = position3(:,1:end);
end

% Find the same minimum step time for three angles
minsteps = min([size(position1,2), size(position2,2), size(position3,2)]);

% Save the six positions into a matrix
X = zeros(6,minsteps);
X(1,:) = position1(1,1:minsteps);
X(2,:) = position1(2,1:minsteps);
X(3,:) = position2(1,1:minsteps);
X(4,:) = position2(2,1:minsteps);
X(5,:) = position3(1,1:minsteps);
X(6,:) = position3(2,1:minsteps);   

% Plot the height of bucket measured by camera
figure()
plot(X(1,:)), hold on
plot(X(3,:))
plot(X(6,:))
xlabel('Time');
ylabel('Height'); 
title('Height of bucket measured by camera');
legend('Camera 1', 'Camera 2', 'Camera 3');
print(gcf,'-dpng','Figure10.png');

% Adjust the time to [0,10] to get the same length for SVD calculation
t = linspace(0,10,minsteps);

% Find mean of each row of the data and subtract it
X = X - repmat(mean(X, 2), 1, minsteps);

% Calculate SVD
[U, S, V] = svd(X / sqrt(minsteps-1));
lambda = diag(S).^2;

% Plot Percentage of variance in different principal directions
figure()
plot((lambda / sum(lambda) * 100) ,'ro','Linewidth',2)
title('Percentage of variance in different principal directions')
xlabel('principal direction')
ylabel('Percent of variance by SVD %')
print(gcf,'-dpng','Figure11.png');

% Use PCA to reduce he redundancy
Y = U' * X;

% Plot height of bucket by PCA
figure()
plot(t,Y(1,:),t,Y(2,:),t,Y(3,:),t,Y(4,:),t,Y(5,:),t,Y(6,:))
legend('Direction 1','Direction 2','Direction 3','Direction 4','Direction 5','Direction 6')
xlabel('Time')
ylabel('Height of bucket')
title('Height of bucket by PCA')
print(gcf,'-dpng','Figure12.png');











