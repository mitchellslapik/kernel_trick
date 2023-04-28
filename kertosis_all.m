
ss = [];
ks = [];

for batch = 1:3

    for minibatch = 1:10

    load('mnist simple complex/data/mnist.mat');
    
    Set1 = brewermap(6,"Set1");

    start = (batch - 1)*10000 + (minibatch - 1)*1000 + 1;
    ending = (batch - 1)*10000 + minibatch*1000;

    XTrain = XTrain(:, :, :, start:ending);
    
    XTrain_original = XTrain;
    YTrain_original = YTrain;
    
    rainbow = jet(10);
    
    XTrain = double(XTrain);
    XTrain = permute(XTrain, [4 1 2 3]);
    XTrain = reshape(XTrain, [1000 28*28]);
    XTrain = zscore(XTrain);
    
    YTrain = YTrain(1:10000);
    
    YTrain = categorical(YTrain);
    YTest = categorical(YTest);
    
    k = kurtosis(XTrain, 1,'all');
    s = skewness(XTrain, 1, 'all');

    ss = [ss, s];
    ks = [ks, k];

    end

end


subplot(3, 3, 2);

scatter(ss, ks, 100, [0 0 0], 'filled');
hold on; 

alpha(.5);

ax = gca;
ax.FontSize = 14; 
box off;

subplot(3, 3, 5);

scatter(ss, ks, 100, [0 0 0], 'filled');
hold on; 

alpha(.5);

ax = gca;
ax.FontSize = 14; 
box off;

subplot(3, 3, 8);

scatter(ss, ks, 100, [0 0 0], 'filled');
hold on; 

alpha(.5);

ax = gca;
ax.FontSize = 14; 
box off;

%%

ss = [];
ks = [];

for batch = 1:3

    for minibatch = 1:10

    load('mnist simple complex/data/mnist.mat');

    start = (batch - 1)*10000 + (minibatch - 1)*1000 + 1;
    ending = (batch - 1)*10000 + minibatch*1000;

    XTrain = XTrain(:, :, :, start:ending);
    
    XTrain_original = XTrain;
    YTrain_original = YTrain;
    
    rainbow = jet(10);

    XTrain = double(XTrain);

    YTrain = YTrain(1:10000);
    
    YTrain = categorical(YTrain);
    YTest = categorical(YTest);
    
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

    XTrain = reshape(XTrain, [1000 28*28*8]);

    XTrain = zscore(XTrain);

    k = kurtosis(XTrain, 1,'all');
    s = skewness(XTrain, 1, 'all');

    ss = [ss, s];
    ks = [ks, k];

    end

end

subplot(3, 3, 2);

scatter(ss, ks, 100, Set1(2, :), 'filled');
hold on; 

alpha(.5);

ax = gca;
ax.FontSize = 14; 
box off;

%%

ss = [];
ks = [];

for batch = 1:3

    for minibatch = 1:10

    load('mnist simple complex/data/mnist.mat');

    start = (batch - 1)*10000 + (minibatch - 1)*1000 + 1;
    ending = (batch - 1)*10000 + minibatch*1000;

    XTrain = XTrain(:, :, :, start:ending);
    
    XTrain_original = XTrain;
    YTrain_original = YTrain;
    
    rainbow = jet(10);

    XTrain = double(XTrain);

    YTrain = YTrain(1:10000);
    
    YTrain = categorical(YTrain);
    YTest = categorical(YTest);
    
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
        
                  gabor(x, y) = 1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*randn();%cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);
        
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

    XTrain = reshape(XTrain, [1000 28*28*8]);

    XTrain = zscore(XTrain);

    k = kurtosis(XTrain, 1,'all');
    s = skewness(XTrain, 1, 'all');

    ss = [ss, s];
    ks = [ks, k];

    end

end

subplot(3, 3, 5);

scatter(ss, ks, 100, Set1(2, :), 'filled');
hold on; 

alpha(.5);

ax = gca;
ax.FontSize = 14; 
box off;

%%

ss = [];
ks = [];

for batch = 1:3

    for minibatch = 1:10

    load('mnist simple complex/data/mnist.mat');

    start = (batch - 1)*10000 + (minibatch - 1)*1000 + 1;
    ending = (batch - 1)*10000 + minibatch*1000;

    XTrain = XTrain(:, :, :, start:ending);
    
    XTrain_original = XTrain;
    YTrain_original = YTrain;
    
    rainbow = jet(10);

    XTrain = double(XTrain);

    YTrain = YTrain(1:10000);
    
    YTrain = categorical(YTrain);
    YTest = categorical(YTest);
    
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
        
                  gabor(x, y) = randn();%1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*randn();%cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);
        
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

    XTrain = reshape(XTrain, [1000 28*28*8]);

    XTrain = zscore(XTrain);

    k = kurtosis(XTrain, 1,'all');
    s = skewness(XTrain, 1, 'all');

    ss = [ss, s];
    ks = [ks, k];

    end

end

subplot(3, 3, 8);

scatter(ss, ks, 100, Set1(2, :), 'filled');
hold on; 

alpha(.5);

ax = gca;
ax.FontSize = 14; 
box off;


%%
ss = [];
ks = [];

for batch = 1:3

    for minibatch = 1:10
    
    load('mnist simple complex/data/mnist.mat');
    

    start = (batch - 1)*10000 + (minibatch - 1)*1000 + 1;
    ending = (batch - 1)*10000 + minibatch*1000;

    XTrain = XTrain(:, :, :, start:ending);
    
    XTrain_original = XTrain;
    YTrain_original = YTrain;
    
    rainbow = jet(10);

    XTrain = double(XTrain);

    YTrain = YTrain(1:10000);
    
    YTrain = categorical(YTrain);
    YTest = categorical(YTest);
    
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

    XTrain = reshape(XTrain, [1000 28*28*8]);

    XTrain = zscore(XTrain);

    k = kurtosis(XTrain, 1,'all');
    s = skewness(XTrain, 1, 'all');

    ss = [ss, s];
    ks = [ks, k];

    end

end

subplot(3, 3, 2);

scatter(ss, ks, 100, Set1(1, :), 'filled');
hold on; 

alpha(.5);

ax = gca;
ax.FontSize = 14; 
box off;

%%

ss = [];
ks = [];

for batch = 1:3

    for minibatch = 1:10
    
    load('mnist simple complex/data/mnist.mat');
    

    start = (batch - 1)*10000 + (minibatch - 1)*1000 + 1;
    ending = (batch - 1)*10000 + minibatch*1000;

    XTrain = XTrain(:, :, :, start:ending);
    
    XTrain_original = XTrain;
    YTrain_original = YTrain;
    
    rainbow = jet(10);

    XTrain = double(XTrain);

    YTrain = YTrain(1:10000);
    
    YTrain = categorical(YTrain);
    YTest = categorical(YTest);
    
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
        
                  gabor(x, y) = 1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*randn();%cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);
        
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
        
                  gabor(x, y) = 1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*randn();%cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);
        
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

    XTrain = reshape(XTrain, [1000 28*28*8]);

    XTrain = zscore(XTrain);

    k = kurtosis(XTrain, 1,'all');
    s = skewness(XTrain, 1, 'all');

    ss = [ss, s];
    ks = [ks, k];

    end

end

subplot(3, 3, 5);

scatter(ss, ks, 100, Set1(1, :), 'filled');
hold on; 

alpha(.5);

ax = gca;
ax.FontSize = 14; 
box off;

%%

ss = [];
ks = [];

for batch = 1:3

    for minibatch = 1:10
    
    load('mnist simple complex/data/mnist.mat');
    

    start = (batch - 1)*10000 + (minibatch - 1)*1000 + 1;
    ending = (batch - 1)*10000 + minibatch*1000;

    XTrain = XTrain(:, :, :, start:ending);
    
    XTrain_original = XTrain;
    YTrain_original = YTrain;
    
    rainbow = jet(10);

    XTrain = double(XTrain);

    YTrain = YTrain(1:10000);
    
    YTrain = categorical(YTrain);
    YTest = categorical(YTest);
    
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
        
                  gabor(x, y) = randn();%1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);
        
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
        
                  gabor(x, y) = randn();%1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);
        
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

    XTrain = reshape(XTrain, [1000 28*28*8]);

    XTrain = zscore(XTrain);

    k = kurtosis(XTrain, 1,'all');
    s = skewness(XTrain, 1, 'all');

    ss = [ss, s];
    ks = [ks, k];

    end

end

subplot(3, 3, 8);

scatter(ss, ks, 100, Set1(1, :), 'filled');
hold on; 

alpha(.5);

ax = gca;
ax.FontSize = 14; 
box off;

%%
ss = [];
ks = [];


for batch = 1:3

    for minibatch = 1:10

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
    XTest = test.data;
    YTest = categorical(test.labels);

    start = (minibatch - 1)*1000 + 1;
    ending = minibatch*1000;

    XTrain = XTrain(start:ending, :);
    
    XTrain = reshape(XTrain, [1000, 32, 32, 3]);

    XTrain_gray = zeros(1000, 32, 32, 1);
    
    for n = 1:1000
    
        picture = XTrain(n, :, :, :);
        picture = squeeze(picture);
        picture = double(picture);
    
        XTrain_gray(n, :, :, :) = rgb2gray(picture/255);
    
    end
    
    XTrain_gray = reshape(XTrain_gray, [1000, 32*32]);

    XTrain_gray = zscore(XTrain_gray);
    
    k = kurtosis(XTrain, 1,'all');
    s = skewness(XTrain, 1, 'all');

    ss = [ss, s];
    ks = [ks, k];


    end

end

subplot(3, 3, 3);

scatter(ss, ks, 100, [0 0 0], 'filled');
hold on; 

alpha(.5);

ax = gca;
ax.FontSize = 14; 
box off;

subplot(3, 3, 6);

scatter(ss, ks, 100, [0 0 0], 'filled');
hold on; 

alpha(.5);

ax = gca;
ax.FontSize = 14; 
box off;

subplot(3, 3, 9);

scatter(ss, ks, 100, [0 0 0], 'filled');
hold on; 

alpha(.5);

ax = gca;
ax.FontSize = 14; 
box off;
%%
ss = [];
ks = [];

for batch = 1:3

    for minibatch = 1:10

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
    

    start = (minibatch - 1)*1000 + 1;
    ending = minibatch*1000;


    XTrain = XTrain(start:ending, :);
    

    XTrain = reshape(XTrain, [1000, 32, 32, 3]);
 
    XTrain_gray = zeros(1000, 32, 32, 1);
    
    for n = 1:1000
    
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

    XTrain_gray = reshape(XTrain_gray, [1000, 32*32*8]);

    XTrain = zscore(XTrain_gray);
    
    k = kurtosis(XTrain, 1,'all');
    s = skewness(XTrain, 1, 'all');

    ss = [ss, s];
    ks = [ks, k];

    end

end

subplot(3, 3, 3)

scatter(ss, ks, 100, Set1(2, :), 'filled');
hold on; 

alpha(.5);

ax = gca;
ax.FontSize = 14; 
box off;

%%

ss = [];
ks = [];

for batch = 1:3

    for minibatch = 1:10

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
    

    start = (minibatch - 1)*1000 + 1;
    ending = minibatch*1000;


    XTrain = XTrain(start:ending, :);
    

    XTrain = reshape(XTrain, [1000, 32, 32, 3]);
 
    XTrain_gray = zeros(1000, 32, 32, 1);
    
    for n = 1:1000
    
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
        
                  gabor(x, y) = 1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*randn();%cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);
        
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

    XTrain_gray = reshape(XTrain_gray, [1000, 32*32*8]);

    XTrain = zscore(XTrain_gray);
    
    k = kurtosis(XTrain, 1,'all');
    s = skewness(XTrain, 1, 'all');

    ss = [ss, s];
    ks = [ks, k];

    end

end

subplot(3, 3, 6)

scatter(ss, ks, 100, Set1(2, :), 'filled');
hold on; 

alpha(.5);

ax = gca;
ax.FontSize = 14; 
box off;

%%

ss = [];
ks = [];

for batch = 1:3

    for minibatch = 1:10

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
    

    start = (minibatch - 1)*1000 + 1;
    ending = minibatch*1000;


    XTrain = XTrain(start:ending, :);
    

    XTrain = reshape(XTrain, [1000, 32, 32, 3]);
 
    XTrain_gray = zeros(1000, 32, 32, 1);
    
    for n = 1:1000
    
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
        
                  gabor(x, y) = randn();%1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);
        
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

    XTrain_gray = reshape(XTrain_gray, [1000, 32*32*8]);

    XTrain = zscore(XTrain_gray);
    
    k = kurtosis(XTrain, 1,'all');
    s = skewness(XTrain, 1, 'all');

    ss = [ss, s];
    ks = [ks, k];

    end

end

subplot(3, 3, 9)

scatter(ss, ks, 100, Set1(2, :), 'filled');
hold on; 

alpha(.5);

ax = gca;
ax.FontSize = 14; 
box off;


%%
ss = [];
ks = [];

for batch = 1:3

    for minibatch = 1:10

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


    start = (minibatch - 1)*1000 + 1;
    ending = minibatch*1000;


    XTrain = XTrain(start:ending, :);
    
    XTrain = reshape(XTrain, [1000, 32, 32, 3]);

    XTrain_gray = zeros(1000, 32, 32, 1);
    
    for n = 1:1000
    
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

    XTrain_gray = reshape(XTrain_gray, [1000, 32*32*8]);

    XTrain = zscore(XTrain_gray);
    
    k = kurtosis(XTrain, 1,'all');
    s = skewness(XTrain, 1, 'all');

    ss = [ss, s];
    ks = [ks, k];

    end

end

subplot(3, 3, 3);

scatter(ss, ks, 100, Set1(1, :), 'filled');
hold on; 

alpha(.5);

ax = gca;
ax.FontSize = 14; 
box off;
%%
ss = [];
ks = [];

for batch = 1:3

    for minibatch = 1:10

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


    start = (minibatch - 1)*1000 + 1;
    ending = minibatch*1000;


    XTrain = XTrain(start:ending, :);
    
    XTrain = reshape(XTrain, [1000, 32, 32, 3]);

    XTrain_gray = zeros(1000, 32, 32, 1);
    
    for n = 1:1000
    
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
        
                  gabor(x, y) = 1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*randn();%cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);
        
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
        
                  gabor(x, y) = 1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*randn();%cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);
        
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

    XTrain_gray = reshape(XTrain_gray, [1000, 32*32*8]);

    XTrain = zscore(XTrain_gray);
    
    k = kurtosis(XTrain, 1,'all');
    s = skewness(XTrain, 1, 'all');

    ss = [ss, s];
    ks = [ks, k];

    end

end

subplot(3, 3, 6);

scatter(ss, ks, 100, Set1(1, :), 'filled');
hold on; 

alpha(.5);

ax = gca;
ax.FontSize = 14; 
box off;

%%
ss = [];
ks = [];

for batch = 1:3

    for minibatch = 1:10

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


    start = (minibatch - 1)*1000 + 1;
    ending = minibatch*1000;


    XTrain = XTrain(start:ending, :);
    
    XTrain = reshape(XTrain, [1000, 32, 32, 3]);

    XTrain_gray = zeros(1000, 32, 32, 1);
    
    for n = 1:1000
    
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
        
                  gabor(x, y) = randn();%1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);
        
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
        
                  gabor(x, y) = randn();%1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);
        
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

    XTrain_gray = reshape(XTrain_gray, [1000, 32*32*8]);

    XTrain = zscore(XTrain_gray);
    
    k = kurtosis(XTrain, 1,'all');
    s = skewness(XTrain, 1, 'all');

    ss = [ss, s];
    ks = [ks, k];

    end

end

subplot(3, 3, 9);

scatter(ss, ks, 100, Set1(1, :), 'filled');
hold on; 

alpha(.5);

ax = gca;
ax.FontSize = 14; 
box off;



%%

set(gcf,'color','w');

gabor = load('real_gabor.mat');

subplot(3, 3, 1); 

imshow(gabor.gabor); 

title('filters', 'FontSize', 14', 'FontWeight', 'normal');

ylabel('gabor', 'FontSize', 14)

gabor = load('random_gabor.mat');

subplot(3, 3, 4); 

imshow(gabor.gabor); 

ylabel('local noise', 'FontSize', 14)

gabor = load('random.mat');

subplot(3, 3, 7); 

imshow(gabor.gabor); 

ylabel('global noise', 'FontSize', 14)

subplot(3, 3, 2); 
title('mnist', 'FontSize', 14, 'FontWeight', 'normal');
ylabel('kurtosis', 'FontSize', 14)
xticks([]);

subplot(3, 3, 5); 
ylabel('kurtosis', 'FontSize', 14)
xticks([]);

subplot(3, 3, 8); 
ylabel('kurtosis', 'FontSize', 14)
xlabel('skewness', 'FontSize', 14);

subplot(3, 3, 3); 
title('cifar10', 'FontSize', 14, 'FontWeight', 'normal');
xticks([]);

xlim([0 2]);
ylim([0 10]);

subplot(3, 3, 6); 

xlim([0 2]);
ylim([0 10]);
xticks([]);

subplot(3, 3, 9); 

xlim([0 2]);
ylim([0 10]);

xlabel('skewness', 'FontSize', 14);

set(gcf,'color','w');
