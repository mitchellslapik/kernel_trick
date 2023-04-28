hs = zeros(1, 101);

mnist_index = 2;
cifar_index =8;

for batch = 1:1

    load('mnist simple complex/data/mnist.mat');
    
    Set1 = brewermap(6,"Set1");
    
    XTrain_original = XTrain;
    YTrain_original = YTrain;
    
    rainbow = jet(10);
    
    start = (batch - 1)*10000 + 1;
    ending = batch * 10000;

    XTrain = XTrain(:, :, :, mnist_index);
    XTrain = double(XTrain);
    XTrain = permute(XTrain, [4 1 2 3]);
    XTrain = reshape(XTrain, [1 28*28]);
    XTrain = zscore(XTrain);
    
    YTrain = YTrain(1:10000);
    
    YTrain = categorical(YTrain);
    YTest = categorical(YTest);
    
    h = hist(XTrain(:), -5:.1:5, 'Normalization','probability');
    h = h/sum(h);

    hs = [hs; h];

end

figure;

subplot(2, 3, 1);

XTrain = (XTrain - min(XTrain)) ./ (max(XTrain) - min(XTrain));

XTrain = reshape(XTrain, [28 28]);
XTrain = imresize(XTrain, 10);

imshow(XTrain);


%%
hs = zeros(1, 101);

for batch = 1:1

    load('mnist simple complex/data/mnist.mat');
    
    XTrain_original = XTrain;
    YTrain_original = YTrain;
    
    rainbow = jet(10);
    
    start = (batch - 1)*10000 + 1;
    ending = batch * 10000;

    XTrain = XTrain(:, :, :, mnist_index);
    XTrain = double(XTrain);
    
    gabors = zeros(56, 56);
    
    for angle_index = 1:8
        
        gabor = zeros(56, 56);
        
        x_std = 5;
        y_std = 5;
        k = .5;
        offset = 0;
        angle = angle_index*pi/8;
        
        for x = 1:56
        
            x_coordinate = x - 28;
        
             for y = 1:56
        
                 y_coordinate = y - 28;
        
                  gabor(x, y) = 1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);
        
             end
        
        end
    
        gabors = cat(3, gabors, gabor);
    
    end
    
    gabors = gabors(:, :, 2:end);
    
    XTrain = convn(XTrain, gabors);
    XTrain = XTrain(28:55, 28:55, :, :);

    XTrain = zscore(XTrain, 0, 'all');

    XTrain = 1./(1 + exp(-XTrain));

    XTrain = permute(XTrain, [4 1 2 3]);

    XTrain = reshape(XTrain, [1 28*28*8]);
    
    XTrain = zscore(XTrain);

    h = hist(XTrain(:), -5:.1:5, 'Normalization','probability');
    h = h/sum(h);

    hs = [hs; h];

end

XTrain = XTrain(28*28*3 +1: 28*28*4);

XTrain = (XTrain - min(XTrain)) ./ (max(XTrain) - min(XTrain));

XTrain = reshape(XTrain, [28 28]);

subplot(2, 3, 2);
XTrain = imresize(XTrain, 10);
imshow(XTrain);


%%
hs = zeros(1, 101);

for batch = 1:1
    
    load('mnist simple complex/data/mnist.mat');
    
    XTrain_original = XTrain;
    YTrain_original = YTrain;
    
    rainbow = jet(10);
    
    start = (batch - 1)*10000 + 1;
    ending = batch * 10000;

    XTrain = XTrain(:, :, :, mnist_index);
    XTrain = double(XTrain);


    gabors = zeros(56, 56);
    
    for angle = 1:8
        
        gabor = zeros(56, 56);
        
        x_std = 5;
        y_std = 5;
        k = .5;
        offset = 0;
        angle = angle*pi/8;
        
        for x = 1:56
        
            x_coordinate = x - 28;
        
             for y = 1:56
        
                 y_coordinate = y - 28;
        
                  gabor(x, y) = 1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);
        
             end
        
        end
    
        gabors = cat(3, gabors, gabor);
    
    end
    
    gabors = gabors(:, :, 2:end);
    
    gabors_offset = zeros(56, 56);
    
    for angle = 1:8
        
        gabor = zeros(56, 56);
        
        x_std = 5;
        y_std = 5;
        k = .5;
        offset = pi/2;
        angle = angle*pi/8;
        
        for x = 1:56
        
            x_coordinate = x - 28;
        
             for y = 1:56
        
                 y_coordinate = y - 28;
        
                  gabor(x, y) = 1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);
        
             end
        
        end
    
        gabors_offset = cat(3, gabors_offset, gabor);
    
    end
    
    gabors_offset = gabors_offset(:, :, 2:end);
    
    XTrain_offset = convn(XTrain, gabors_offset);
    XTrain_offset = XTrain_offset(28:55, 28:55, :, :);
    
    XTrain = convn(XTrain, gabors);
    XTrain = XTrain(28:55, 28:55, :, :);
    
    XTrain = XTrain.^2 + XTrain_offset.^2;

    XTrain = zscore(XTrain, 0, 'all');

    XTrain = 1./(1 + exp(-XTrain));

    XTrain = permute(XTrain, [4 1 2 3]);

    XTrain = reshape(XTrain, [1 28*28*8]);

    XTrain = zscore(XTrain);

    h = hist(XTrain(:), -5:.1:5, 'Normalization','probability');
    h = h/sum(h);

    hs = [hs; h];

end


XTrain = XTrain(28*28*3 +1: 28*28*4);
XTrain = (XTrain - min(XTrain)) ./ (max(XTrain) - min(XTrain));

XTrain = reshape(XTrain, [28 28]);

subplot(2, 3, 3);
XTrain = imresize(XTrain, 10);
imshow(XTrain);


%%
hs = zeros(1, 101);

for batch = 1:1

    if batch < 6
    
        train = load(['cifar simple complex/data/data_batch_'  num2str(batch) '.mat']);
        test = load('cifar simple complex/data/test_batch.mat');

    else
    
        train = load('cifar simple complex/data/test_batch.mat');
        test = load('cifar simple complex/data/test_batch.mat');

    end

    rainbow = jet(10);
    
    XTrain = train.data;
    YTrain = categorical(train.labels);

    XTrain = XTrain(cifar_index, :);
    XTrain = reshape(XTrain, [1, 32, 32, 3]);

    XTrain_gray = zeros(1, 32, 32, 1);
    
    for n = 1:1
    
        picture = XTrain(n, :, :, :);
        picture = squeeze(picture);
        picture = double(picture);
    
        XTrain_gray(n, :, :, :) = rgb2gray(picture/255);
    
    end
    

    XTrain_gray = reshape(XTrain_gray, [1, 32*32]);

    XTrain_gray = zscore(XTrain_gray);
    
    h = hist(XTrain_gray(:), -5:.1:5, 'Normalization','probability');
    h = h/sum(h);

    hs = [hs; h];

end

subplot(2, 3, 4);

XTrain_gray = (XTrain_gray - min(XTrain_gray)) ./ (max(XTrain_gray) - min(XTrain_gray));

XTrain_gray = reshape(XTrain_gray, [32 32]);

XTrain_gray = permute(XTrain_gray, [2 1]);

XTrain_gray = imresize(XTrain_gray, 10);

imshow(XTrain_gray);

%%
hs = zeros(1, 101);

for batch = 1:1

    if batch < 6
    
        train = load(['cifar simple complex/data/data_batch_' num2str(batch) '.mat']);
        test = load('cifar simple complex/data/test_batch.mat');

    else
    
        train = load('cifar simple complex/data/test_batch.mat');
        test = load('cifar simple complex/data/test_batch.mat');

    end
    rainbow = jet(10);
    
    XTrain = train.data;

    XTrain = XTrain(cifar_index, :);
    XTrain = reshape(XTrain, [1, 32, 32, 3]);
    
    XTrain_gray = zeros(1, 32, 32, 1);
    
    for n = 1:1
    
        picture = XTrain(n, :, :, :);
        picture = squeeze(picture);
        picture = double(picture);
    
        XTrain_gray(n, :, :, :) = rgb2gray(picture/255);
    
    end
    
    XTrain_gray = permute(XTrain_gray, [3 2 4 1]);

    gabors = zeros(64, 64);
    
    for angle_index = 1:8
        
        gabor = zeros(64, 64);
        
        x_std = 5;
        y_std = 5;
        k = .5;
        offset = 0;
        angle = angle_index*pi/8;
        
        for x = 1:64
        
            x_coordinate = x - 32;
        
             for y = 1:64
        
                 y_coordinate = y - 32;
        
                  gabor(x, y) = 1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);
        
             end
        
        end
    
        gabors = cat(3, gabors, gabor);
    
    end
    
    gabors = gabors(:, :, 2:end);
    
    XTrain_gray = convn(XTrain_gray, gabors);
    XTrain_gray = XTrain_gray(32:63, 32:63, :, :);
    
    XTrain_gray = zscore(XTrain_gray, 0, 'all');

    XTrain_gray = 1./(1 + exp(-XTrain_gray));

    XTrain_gray = permute(XTrain_gray, [4 1 2 3]);

    XTrain_gray = reshape(XTrain_gray, [1, 32*32*8]);

    h = hist(XTrain(:), -5:.1:5, 'Normalization','probability');
    h = h/sum(h);

    hs = [hs; h];

end

subplot(2, 3, 5);

XTrain_gray = XTrain_gray(32*32*7 +1: 32*32*8);

XTrain_gray = (XTrain_gray - min(XTrain_gray)) ./ (max(XTrain_gray) - min(XTrain_gray));

XTrain_gray = reshape(XTrain_gray, [32 32]);

XTrain_gray = permute(XTrain_gray, [2 1]);

XTrain_gray = imresize(XTrain_gray, 10);

imshow(XTrain_gray);



%%
hs = zeros(1, 101);

for batch = 1:1

    if batch < 6
    
        train = load(['cifar simple complex/data/data_batch_' num2str(batch) '.mat']);
        test = load('cifar simple complex/data/test_batch.mat');

    else
    
        train = load('cifar simple complex/data/test_batch.mat');
        test = load('cifar simple complex/data/test_batch.mat');

    end
    rainbow = jet(10);
    
    XTrain = train.data;
    YTrain = categorical(train.labels);
    XTest = test.data;
    YTest = categorical(test.labels);

    XTrain = XTrain(cifar_index, :);
    
    XTrain = reshape(XTrain, [1, 32, 32, 3]);

    XTrain_gray = zeros(1, 32, 32, 1);
    
    for n = 1:1
    
        picture = XTrain(n, :, :, :);
        picture = squeeze(picture);
        picture = double(picture);
    
        XTrain_gray(n, :, :, :) = rgb2gray(picture/255);
    
    end
    
    XTrain_gray = permute(XTrain_gray, [3 2 4 1]);

    gabors = zeros(64, 64);
    
    for angle_index = 1:8
        
        gabor = zeros(64, 64);
        
        x_std = 5;
        y_std = 5;
        k = .5;
        offset = 0;
        angle = angle_index*pi/8;
        
        for x = 1:64
        
            x_coordinate = x - 32;
        
             for y = 1:64
        
                 y_coordinate = y - 32;
        
                  gabor(x, y) = 1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);
        
             end
        
        end
    
        gabors = cat(3, gabors, gabor);
    
    end
    
    gabors = gabors(:, :, 2:end);
    
    gabors_offset = zeros(64, 64);
    
    for angle = 1:8
        
        gabor = zeros(64, 64);
        
        x_std = 5;
        y_std = 5;
        k = .5;
        offset = pi/2;
        angle = angle*pi/8;
        
        for x = 1:64
        
            x_coordinate = x - 32;
        
             for y = 1:64
        
                 y_coordinate = y - 32;
        
                  gabor(x, y) = 1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);
        
             end
        
        end
    
        gabors_offset = cat(3, gabors_offset, gabor);
    
    end
    
    gabors_offset = gabors_offset(:, :, 2:end);
    
    XTrain_gray_offset = convn(XTrain_gray, gabors_offset);
    XTrain_gray_offset = XTrain_gray_offset(32:63, 32:63, :, :);
    
    XTrain_gray = convn(XTrain_gray, gabors);
    XTrain_gray = XTrain_gray(32:63, 32:63, :, :);

    XTrain_gray = XTrain_gray.^2 + XTrain_gray_offset.^2;

    XTrain_gray = zscore(XTrain_gray, 0, 'all');

    XTrain_gray = 1./(1 + exp(-XTrain_gray));

    XTrain_gray = permute(XTrain_gray, [4 1 2 3]);

    XTrain_gray = reshape(XTrain_gray, [1, 32*32*8]);


    h = hist(XTrain(:), -5:.1:5, 'Normalization','probability');
    h = h/sum(h);

    hs = [hs; h];

end

subplot(2, 3, 6);

XTrain_gray = XTrain_gray(32*32*7 +1: 32*32*8);


XTrain_gray = (XTrain_gray - min(XTrain_gray)) ./ (max(XTrain_gray) - min(XTrain_gray));

XTrain_gray = reshape(XTrain_gray, [32 32]);

XTrain_gray = permute(XTrain_gray, [2 1]);

XTrain_gray = imresize(XTrain_gray, 10);
imshow(XTrain_gray);



%%

set(gcf,'color','w');

subplot(2, 3, 1)
title('pixels', 'FontSize', 14, 'FontWeight', 'normal');

subplot(2, 3, 2)
title('simple cells', 'FontSize', 14, 'FontWeight', 'normal');

subplot(2, 3, 3)
title('complex cells', 'FontSize', 14, 'FontWeight', 'normal');

subplot(2, 3, 1)
ylabel('mnist', 'FontSize', 14, 'FontWeight', 'normal');

subplot(2, 3, 4)
ylabel('cifar10', 'FontSize', 14, 'FontWeight', 'normal');
