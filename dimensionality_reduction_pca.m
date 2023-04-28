train = load('data/data_batch_1.mat');
test = load('data/test_batch.mat');

rainbow = jet(10);

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

XTrain_gray = reshape(XTrain_gray, [10000, 32*32]);
XTest_gray = reshape(XTest_gray, [10000, 32*32]);

% XTrain_gray = permute(XTrain_gray, [3 2 4 1]);
% XTest_gray = permute(XTest_gray, [3 2 4 1]);

[coeffs, scores, latent] = pca(XTrain_gray);

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