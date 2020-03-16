clc;
clear all;
close all;

%load audio file
load handel;

v = y';
% Play back audio
%p8 = audioplayer(v,Fs);
%playblocking(p8);

% Read the basic terms, calculate time signal, and plot in the upper panel
% of figure 1
v = v(1:end-1);
n = length(v);
L = (n-1) / Fs;
t1=linspace(0,L,n+1); t=t1(1:n);
k=(2*pi/L)*[0:n/2-1 -n/2:-1]; 
ks=fftshift(k);
figure(1);
subplot(2,1,1)
plot(t,v);
xlabel('Time [sec]');
ylabel('Amplitude');
title('Signal of Interest, v(n)');

% Calculate Fourier transform for the time-frequency, and plot in the lower 
% panel of figure 1
vt = fft(v);
subplot(2,1,2);
plot(ks, abs(fftshift(vt)) / max(abs(vt)));
xlabel('Frequency (k)');
ylabel('FFT(v)');
title('FFT of v');
drawnow;
print(gcf,'-dpng','Figure1.png');

% Set width and number of steps, calculate time slide
width = 2; 
numstep = 20;  
tslide = linspace(0,t(end-1),numstep); 
spec = zeros(length(tslide),n); 

% Create Gabor time filter for three different Gaussian filters: 1:
% Gaussian, 2: Mexican hat wevelet, and 3: step-function (Shannon) 
filter = {@(x) exp(-width*(x).^2), @(x)(1-(x/width).^2).*exp(-((x/width).^2)/2), ... 
    @(x) (x>-width/2 & x< width/2)};
for j=1:length(tslide)
g = filter{1}(t-tslide(j)); % Gabor filter (choose 1-3 for three different filters)
vg = g.*v; % Apply Gabor filter
vgt = fft(vg); % Take fft of filtered data
spec(j,:) = abs(fftshift(vgt)); % Store fft in spectrogram

% plot Gabor transforms
%figure(5);
%subplot(3,1,1), plot(t,v,t,g,'r'), title('Time signal and Gabor time filter'), ... 
    %legend('v','Gabor time filter'), xlabel('time (t)'), ylabel('v(t), g(t)');
%subplot(3,1,2), plot(t,vg), title('Gabor time filter * Time signal'), ...
    %xlabel('time (t)'), ylabel('v(t)g(t)');
%subplot(3,1,3), plot(ks, abs(fftshift(vgt))/max(abs(vgt))), ...
    %title('Gabor transform of vg'), xlabel('frequency[Hz]'), ...
    %ylabel('FFT(vg)');
%drawnow;
%print(gcf,'-dpng','Figure5.png');
end


% Plot spectrogram
figure(10);
pcolor(tslide,ks,spec.'), shading interp
colormap('hot') 
xlabel('Time [sec]'), ylabel('Frequency [Hz]');
title('Spectrogram of Gabor Transform');
print(gcf,'-dpng','Figure10.png');
