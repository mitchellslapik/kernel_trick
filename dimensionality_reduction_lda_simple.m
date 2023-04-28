train = load('data/data_batch_1.mat');
test = load('data/test_batch.mat');

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

figure;

for number = 1:10
    
    number_indexes = double(YTrain) == number;

    number_scores = scores(number_indexes, :);

    scatter(number_scores(:, 1), number_scores(:, 2), 10, rainbow(number, :), 'filled');
    alpha(.5);
    hold on;

end

xlabel('PC1', 'FontSize', 14)
ylabel('PC2', 'FontSize', 14)

set(gcf,'color','w');
box off;

ax = gca;
ax.FontSize = 14;

figure;

for number = 1:10
    
    number_indexes = double(YTrain) == number;

    number_scores = scores(number_indexes, :);

    scatter3(number_scores(:, 1), number_scores(:, 2), number_scores(:, 3), 10, rainbow(number, :), 'filled');
    alpha(.5);
    hold on;

end

xlabel('PC1', 'FontSize', 14);
ylabel('PC2', 'FontSize', 14);
zlabel('PC3', 'FontSize', 14);

set(gcf,'color','w');
box off;
grid off

ax = gca;
ax.FontSize = 14;