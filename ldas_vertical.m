load('mnist simple complex/data/mnist.mat');

XTrain_original = XTrain;
YTrain_original = YTrain;

rainbow = jet(10);

XTrain = XTrain(:, :, :, 1:10000);
XTrain = double(XTrain);
XTrain = permute(XTrain, [4 1 2 3]);
XTrain = reshape(XTrain, [10000 28*28]);

XTrain_std = std(XTrain);
XTrain(:, XTrain_std == 0) = [];

YTrain = YTrain(1:10000);

YTrain = categorical(YTrain);
YTest = categorical(YTest);

Mdl = fitcdiscr(XTrain, YTrain);
[W, LAMBDA] = eig(Mdl.BetweenSigma, Mdl.Sigma);
lambda = diag(LAMBDA);
[lambda, SortOrder] = sort(lambda, 'descend');
W = W(:, SortOrder);
scores = XTrain*W;

scores = real(scores);

figure('Position', [300 300 800 1000]);

subplot(3, 2, 1);

for number = 1:10
    
    number_indexes = double(YTrain) == number - 1;

    number_scores = scores(number_indexes, :);

    scatter(number_scores(:, 1), number_scores(:, 2), 10, rainbow(number, :), 'filled');
    alpha(.5);
    hold on;

end

xlabel('ld1', 'FontSize', 14)
ylabel('ld2', 'FontSize', 14)

set(gcf,'color','w');
box off;

ax = gca;
ax.FontSize = 14;

%%
load('mnist simple complex/data/mnist.mat');

XTrain_original = XTrain;
YTrain_original = YTrain;

rainbow = jet(10);

XTrain = XTrain(:, :, :, 1:10000);
XTrain = double(XTrain);
XTest = XTest(:, :, :, 1:10000);
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

XTest = convn(XTest, gabors);
XTest = XTest(28:55, 28:55, :, :);

XTrain = zscore(XTrain, 0, 'all');
XTest = zscore(XTest, 0, 'all');

XTrain = 1./(1 + exp(-XTrain));
XTest = 1./(1 + exp(-XTest));

XTrain = permute(XTrain, [4 1 2 3]);
XTest = permute(XTest, [4 1 2 3]);

XTrain = reshape(XTrain, [10000 28*28*8]);
XTest = reshape(XTest, [10000 28*28*8]);

[coeffs, scores, latent] = pca(XTrain);

XTrain = scores(:, 1:10);

Mdl = fitcdiscr(XTrain, YTrain);
[W, LAMBDA] = eig(Mdl.BetweenSigma, Mdl.Sigma);
lambda = diag(LAMBDA);
[lambda, SortOrder] = sort(lambda, 'descend');
W = W(:, SortOrder);
scores = XTrain*W;

scores = real(scores);

subplot(3, 2, 3);

for number = 1:10
    
    number_indexes = double(YTrain) == number - 1;

    number_scores = scores(number_indexes, :);

    scatter(number_scores(:, 1), number_scores(:, 2), 10, rainbow(number, :), 'filled');
    alpha(.5);
    hold on;

end

xlabel('ld1', 'FontSize', 14)
ylabel('ld2', 'FontSize', 14)

set(gcf,'color','w');
box off;

ax = gca;
ax.FontSize = 14;

%%

load('mnist simple complex/data/mnist.mat');

XTrain_original = XTrain;
YTrain_original = YTrain;

rainbow = jet(10);

XTrain = XTrain(:, :, :, 1:10000);
XTrain = double(XTrain);
XTest = XTest(:, :, :, 1:10000);
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

XTest_offset = convn(XTest, gabors_offset);
XTest_offset = XTest_offset(28:55, 28:55, :, :);

XTest = convn(XTest, gabors);
XTest = XTest(28:55, 28:55, :, :);

XTrain = XTrain.^2 + XTrain_offset.^2;
XTest = XTest.^2 + XTest_offset.^2;

XTrain = zscore(XTrain, 0, 'all');
XTest = zscore(XTest, 0, 'all');

XTrain = 1./(1 + exp(-XTrain));
XTest = 1./(1 + exp(-XTest));

XTrain = permute(XTrain, [4 1 2 3]);
XTest = permute(XTest, [4 1 2 3]);

XTrain = reshape(XTrain, [10000 28*28*8]);
XTest = reshape(XTest, [10000 28*28*8]);

[coeffs, scores, latent] = pca(XTrain);

XTrain = scores(:, 1:10);

Mdl = fitcdiscr(XTrain, YTrain);
[W, LAMBDA] = eig(Mdl.BetweenSigma, Mdl.Sigma);
lambda = diag(LAMBDA);
[lambda, SortOrder] = sort(lambda, 'descend');
W = W(:, SortOrder);
scores = XTrain*W;

scores = real(scores);

subplot(3, 2, 5);

for number = 1:10
    
    number_indexes = double(YTrain) == number - 1;

    number_scores = scores(number_indexes, :);

    scatter(number_scores(:, 1), number_scores(:, 2), 10, rainbow(number, :), 'filled');
    alpha(.5);
    hold on;

end

xlabel('ld1', 'FontSize', 14)
ylabel('ld2', 'FontSize', 14)

set(gcf,'color','w');
box off;

ax = gca;
ax.FontSize = 14;

%%

train = load('cifar simple complex/data/data_batch_1.mat');
test = load('cifar simple complex/data/test_batch.mat');

rainbow = jet(10);

XTrain = train.data;
YTrain = categorical(train.labels);
XTest = test.data;
YTest = categorical(test.labels);

XTrain = reshape(XTrain, [10000, 32, 32, 3]);
XTest = reshape(XTest, [10000, 32, 32, 3]);

XTrain_gray = zeros(10000, 32, 32, 1);

for n = 1:10000

    picture = XTrain(n, :, :, :);
    picture = squeeze(picture);
    picture = double(picture);

    XTrain_gray(n, :, :, :) = rgb2gray(picture/255);

end

XTest_gray = zeros(10000, 32, 32, 1);

for n = 1:10000

    picture = XTest(n, :, :, :);
    picture = squeeze(picture);

    XTest_gray(n, :, :, :) = rgb2gray(picture/255);

end

XTrain_gray = reshape(XTrain_gray, [10000, 32*32]);
XTest_gray = reshape(XTest_gray, [10000, 32*32]);

Mdl = fitcdiscr(XTrain_gray, YTrain);
[W, LAMBDA] = eig(Mdl.BetweenSigma, Mdl.Sigma);
lambda = diag(LAMBDA);
[lambda, SortOrder] = sort(lambda, 'descend');
W = W(:, SortOrder);
scores = XTrain_gray*W;

scores = real(scores);

subplot(3, 2, 2);

for number = 1:10
    
    number_indexes = double(YTrain) == number;

    number_scores = scores(number_indexes, :);

    scatter(number_scores(:, 1), number_scores(:, 2), 10, rainbow(number, :), 'filled');
    alpha(.5);
    hold on;

end

xlabel('ld1', 'FontSize', 14)
ylabel('ld2', 'FontSize', 14)

set(gcf,'color','w');
box off;

ax = gca;
ax.FontSize = 14;

%%

train = load('cifar simple complex/data/data_batch_1.mat');
test = load('cifar simple complex/data/test_batch.mat');

rainbow = jet(10);

XTrain = train.data;
YTrain = categorical(train.labels);
XTest = test.data;
YTest = categorical(test.labels);

XTrain = reshape(XTrain, [10000, 32, 32, 3]);
XTest = reshape(XTest, [10000, 32, 32, 3]);

XTrain_gray = zeros(10000, 32, 32, 1);

for n = 1:10000

    picture = XTrain(n, :, :, :);
    picture = squeeze(picture);
    picture = double(picture);

    XTrain_gray(n, :, :, :) = rgb2gray(picture/255);

end

XTest_gray = zeros(10000, 32, 32, 1);

for n = 1:10000

    picture = XTest(n, :, :, :);
    picture = squeeze(picture);

    XTest_gray(n, :, :, :) = rgb2gray(picture/255);

end

XTrain_gray = permute(XTrain_gray, [3 2 4 1]);
XTest_gray = permute(XTest_gray, [3 2 4 1]);

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

XTest_gray = convn(XTest_gray, gabors);
XTest_gray = XTest_gray(32:63, 32:63, :, :);

XTrain_gray = zscore(XTrain_gray, 0, 'all');
XTest_gray = zscore(XTest_gray, 0, 'all');

XTrain_gray = 1./(1 + exp(-XTrain_gray));
XTest_gray = 1./(1 + exp(-XTest_gray));

XTrain_gray = permute(XTrain_gray, [4 1 2 3]);
XTest_gray = permute(XTest_gray, [4 1 2 3]);

XTrain_gray = reshape(XTrain_gray, [10000, 32*32*8]);
XTest_gray = reshape(XTest_gray, [10000, 32*32*8]);

[coeffs, scores, latent] = pca(XTrain_gray);

XTrain_gray = scores(:, 1:10);

Mdl = fitcdiscr(XTrain_gray, YTrain);
[W, LAMBDA] = eig(Mdl.BetweenSigma, Mdl.Sigma);
lambda = diag(LAMBDA);
[lambda, SortOrder] = sort(lambda, 'descend');
W = W(:, SortOrder);
scores = XTrain_gray*W;

scores = real(scores);

subplot(3, 2, 4);

for number = 1:10
    
    number_indexes = double(YTrain) == number;

    number_scores = scores(number_indexes, :);

    scatter(number_scores(:, 1), number_scores(:, 2), 10, rainbow(number, :), 'filled');
    alpha(.5);
    hold on;

end

xlabel('ld1', 'FontSize', 14)
ylabel('ld2', 'FontSize', 14)

set(gcf,'color','w');
box off;

ax = gca;
ax.FontSize = 14;

%%

train = load('cifar simple complex/data/data_batch_1.mat');
test = load('cifar simple complex/data/test_batch.mat');

rainbow = jet(10);

XTrain = train.data;
YTrain = categorical(train.labels);
XTest = test.data;
YTest = categorical(test.labels);

XTrain = reshape(XTrain, [10000, 32, 32, 3]);
XTest = reshape(XTest, [10000, 32, 32, 3]);

XTrain_gray = zeros(10000, 32, 32, 1);

for n = 1:10000

    picture = XTrain(n, :, :, :);
    picture = squeeze(picture);
    picture = double(picture);

    XTrain_gray(n, :, :, :) = rgb2gray(picture/255);

end

XTest_gray = zeros(10000, 32, 32, 1);

for n = 1:10000

    picture = XTest(n, :, :, :);
    picture = squeeze(picture);

    XTest_gray(n, :, :, :) = rgb2gray(picture/255);

end

XTrain_gray = permute(XTrain_gray, [3 2 4 1]);
XTest_gray = permute(XTest_gray, [3 2 4 1]);

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

XTest_gray_offset = convn(XTest_gray, gabors_offset);
XTest_gray_offset = XTest_gray_offset(32:63, 32:63, :, :);

XTest_gray = convn(XTest_gray, gabors);
XTest_gray = XTest_gray(32:63, 32:63, :, :);

XTrain_gray = XTrain_gray.^2 + XTrain_gray_offset.^2;
XTest_gray = XTest_gray.^2 + XTest_gray_offset.^2;

XTrain_gray = zscore(XTrain_gray, 0, 'all');
XTest_gray = zscore(XTest_gray, 0, 'all');

XTrain_gray = 1./(1 + exp(-XTrain_gray));
XTest_gray = 1./(1 + exp(-XTest_gray));

XTrain_gray = permute(XTrain_gray, [4 1 2 3]);
XTest_gray = permute(XTest_gray, [4 1 2 3]);

XTrain_gray = reshape(XTrain_gray, [10000, 32*32*8]);
XTest_gray = reshape(XTest_gray, [10000, 32*32*8]);

[coeffs, scores, latent] = pca(XTrain_gray);

XTrain_gray = scores(:, 1:10);

Mdl = fitcdiscr(XTrain_gray, YTrain);
[W, LAMBDA] = eig(Mdl.BetweenSigma, Mdl.Sigma);
lambda = diag(LAMBDA);
[lambda, SortOrder] = sort(lambda, 'descend');
W = W(:, SortOrder);
scores = XTrain_gray*W;

scores = real(scores);

subplot(3, 2, 6);

for number = 1:10
    
    number_indexes = double(YTrain) == number;

    number_scores = scores(number_indexes, :);

    scatter(number_scores(:, 1), number_scores(:, 2), 10, rainbow(number, :), 'filled');
    alpha(.5);
    hold on;

end

xlabel('ld1', 'FontSize', 14)
ylabel('ld2', 'FontSize', 14)

set(gcf,'color','w');
box off;

ax = gca;
ax.FontSize = 14;
