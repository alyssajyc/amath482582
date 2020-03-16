clc;
clear all; 
close all; 

% load 9 wav files
[x1,Fs] = audioread('kpopgd.wav');
[x2,Fs] = audioread('kpopwannaone.wav');
[x3,Fs] = audioread('kpopturbo.wav');

[y1,Fs] = audioread('trothong.wav');
[y2,Fs] = audioread('trotpark.wav');
[y3,Fs] = audioread('trotlee.wav');

[z1,Fs] = audioread('rockwang.wav');
[z2,Fs] = audioread('rockcui.wav');
[z3,Fs] = audioread('rockluo.wav');

% add the wav file in same genre together
x = [x1;x2;x3];
y = [y1;y2;y3];
z = [z1;z2;z3];

L_x = length(x);
L_y = length(y);
L_z = length(z);

% Construct training dataset

num_training = 500;
num_test = 100;
number = 2;

accuracy_lda = [];
acuracy_nb = [];
accuracy_bt = [];

for kk = 1:5

    testnum = unidrnd(floor(0.8*L_x));

    % training data
    x_training = x([1:testnum-1, testnum+floor(0.2*L_x):end]);
    y_training = y([1:testnum-1, testnum+floor(0.2*L_y):end]);
    z_training = z([1:testnum-1, testnum+floor(0.2*L_z):end]);

    % test data
    x_test = x([testnum:testnum+floor(0.2*L_x)]);
    y_test = y([testnum:testnum+floor(0.2*L_y)]);
    z_test = z([testnum:testnum+floor(0.2*L_z)]);

    % get training data
    x_train = test_construct(num_training, x_training, length(x_training), Fs, number);
    y_train = test_construct(num_training, y_training, length(y_training), Fs, number);
    z_train = test_construct(num_training,z_training, length(z_training), Fs, number);

    % Plot clips

    clipind = 10000;

    figure(1)
%     subplot(3,1,1)
    plot(0:1/Fs:5,z(clipind:clipind+5*Fs))
    title('z')
%     subplot(3,1,2)
    figure(2)
    plot(0:1/Fs:5,x(clipind:clipind+5*Fs))
    title('x')
%     subplot(3,1,3)
    figure(3)
    plot(0:1/Fs:5,y(clipind:clipind+5*Fs))
    title('y')

    % Construct labels

    labels = [ones(num_training,1);2*ones(num_training,1);3*ones(num_training,1)];

    % Construct total training dataset

    training = abs([x_train';y_train';z_train']);

    % Construct test dataset

    test_labels = [ones(num_test,1);2*ones(num_test,1);3*ones(num_test,1)];

    x_test = test_construct(num_test, x_test, length(x_test), Fs, number);
    y_test = test_construct(num_test, y_test, length(y_test), Fs, number);
    z_test = test_construct(num_test, z_test, length(z_test), Fs, number);

    sample = abs([x_test';y_test';z_test']);

    % Classification

    class = classify(sample, training, labels);
    accuracy = sum(class==test_labels)/length(class);

    Mdl_ctree = fitctree(training, labels);
    class_ctree = predict(Mdl_ctree, sample);
    accuracy_ctree = sum(class_ctree==test_labels)/length(class);

    Mdl_cnb = fitcnb(training, labels);
    class_cnb = predict(Mdl_cnb, sample);
    accuracy_cnb = sum(class_cnb==test_labels)/length(class);
    
    accuracy_lda = [accuracy_lda; accuracy];
    accuracy_bt = [accuracy_bt; accuracy_ctree];
    acuracy_nb = [acuracy_nb; accuracy_cnb];
end    

%

figure(4)
plot(accuracy_lda(1:5), '-');
hold on;
plot(accuracy_bt(1:5), '-');
plot(acuracy_nb(1:5), '-');
legend({'LDA','Binary Tree', 'Naive Bayes'},'Location','southeast');
xlabel('Number of Modes')
ylabel('Accuracy(%)')
title('Test 3')

figure(5)
% subplot(1,3,1)
pstart = 10000;
pend = pstart + 5*Fs;
clip = x(pstart:pend);
spectrogram(clip,gausswin(500),200,[],Fs,'yaxis');
title('k-pop')

figure(6)
% subplot(1,3,2)
pstart = 10000;
pend = pstart + 5*Fs;
clip = y(pstart:pend);
spectrogram(clip,gausswin(500),200,[],Fs,'yaxis');
title('trot')

figure(7)
% subplot(1,3,3)
pstart = 10000;
pend = pstart + 5*Fs;
clip = z(pstart:pend);
spectrogram(clip,gausswin(500),200,[],Fs,'yaxis');
title('Chinese Rock')















