clc;
clear all;
close all;

% load piano - music 1
[y,Fs] = audioread('music2.wav');
tr_recorder=length(y)/Fs; % record time in seconds
v = y.';

% plot figure 1
figure(12);
subplot(2,1,1);
plot((1:length(v))/Fs,v);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Mary had a little lamb (recorder)');

% play back audio
% p8 = audioplayer(y,Fs); 
% playblocking(p8);

% start Gabor Transform
L = tr_recorder;
n = length(v);
t1 = linspace(0,L,n+1);
t = t1(1:n);
k = (2*pi/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);
vt = fft(v);

% Look at frequency components of entire signal
subplot(2,1,2)
plot(ks, abs(fftshift(vt)) / max(abs(vt)));
xlabel('Frequency [Hz]'), ylabel('FFT(v)'), title('FFT of v (recorder)');
print(gcf,'-dpng','Figure12.png');

% Create Gabor filter for spectrogram
figure(14);
vgt_spec=[];
width = 50; % Width of filter
numstep = 200; % Number of time steps to take
tslide = linspace(0,L,numstep); % Time discretization

% Create spectrogram using Gabor filter
for j=1:length(tslide)
g = exp(-width*(t - tslide(j)).^2); % Gabor 
vg = g .* v;
vgt = fft(vg);
vgt_spec=[vgt_spec; abs(fftshift(vgt))];

% plot Gabor transforms
%figure(14);
%subplot(3,1,1), plot(t,v,t,g,'r'), title('Time signal and Gabor time filter'), ... 
    %legend('v','Gabor time filter'), xlabel('time (t)'), ylabel('v(t), g(t)');
%subplot(3,1,2), plot(t,vg), title('Gabor time filter * Time signal'), ...
    %xlabel('time (t)'), ylabel('v(t)g(t)');
%subplot(3,1,3), plot(ks, abs(fftshift(vgt))/max(abs(vgt))), ...
    %title('Gabor transform of vg'), xlabel('frequency[Hz]'), ...
    %ylabel('FFT(vg)');
%drawnow;
%print(gcf,'-dpng','Figure14.png');
end

% Plot relevant portion of spectrogram
figure(16);
pcolor(tslide,ks,log(vgt_spec.')), shading interp;
colormap('hot');
xlabel('Time [sec]'); 
ylabel('Frequency [Hz]');
title('Spectrogram of recorder');
print(gcf,'-dpng','Figure16.png');

% Plot relevant portion of spectrogram
figure(18);
pcolor(tslide,ks,log(vgt_spec.')), shading interp
axis([0 15 1400 2200])
colormap('hot'), xlabel('Time [sec]'), ylabel('Frequency [Hz]'),
title('Log of recorder Spectrogram (zoomed)')
print(gcf,'-dpng','Figure18.png');
