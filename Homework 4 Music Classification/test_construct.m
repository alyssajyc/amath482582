function [trainingset] = test_construct(num, song, length, Fs, number)
    trainingset = [];
    for j = 1:num
        feature = [];
        pstart = unidrnd(length-5*Fs);
        pend = pstart + 5*Fs;
        clip = song(pstart:pend);
        [spec_clip] = spectrogram(clip,gausswin(500),200,[],Fs);
        [u,s,v] = svd(spec_clip,'econ');
        for j = 1:number
            feature = [feature;u(:,j)];
        end
        trainingset = [trainingset, feature]; 
    end
end
