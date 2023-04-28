
figure('Position', [300 300 1000 500]);


mnist_lgn = load('mnist simple complex/svm_on_lgn_gains.mat');
mnist_simple = load('mnist simple complex/svm_on_simple_gains.mat');
mnist_complex = load('mnist simple complex/svm_on_complex_gains.mat');

cifar_lgn = load('cifar simple complex/svm_on_lgn_gains.mat');
cifar_simple = load('cifar simple complex/svm_on_simple_gains.mat');
cifar_complex = load('cifar simple complex/svm_on_complex_gains.mat');

train_acc = zeros(11, 11);
test_acc = zeros(11, 11);
k = zeros(11, 11);

for gain = 1:11

    for threshold = 1:11

        train_acc(gain, threshold) = mnist_lgn.activations(threshold).thresold(gain).gain.trainacc;
        test_acc(gain, threshold) = mnist_lgn.activations(threshold).thresold(gain).gain.testacc;
        k(gain, threshold) = mnist_lgn.activations(threshold).thresold(gain).gain.k;

    end

end

subplot(3, 6, 1);

heatmap(train_acc, 'CellLabelColor','none');

colormap jet;
colorbar off
clim([.2 1]);

%ylabel('gain');

Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
Ax.GridVisible = 'off'

subplot(3, 6, 7);

heatmap(test_acc, 'CellLabelColor','none');

colormap jet;
clim([.2 1]);
colorbar off

Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
Ax.GridVisible = 'off'

%ylabel('gain');

subplot(3, 6, 13);

heatmap(log(k), 'CellLabelColor','none');
clim([2 8]);

colormap jet;
colorbar off

Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
Ax.GridVisible = 'off'

%xlabel('threshold');

%ylabel('gain');

train_acc = zeros(11, 11);
test_acc = zeros(11, 11);
k = zeros(11, 11);

for gain = 1:11

    for threshold = 1:11

        train_acc(gain, threshold) = mnist_simple.activations(threshold).thresold(gain).gain.trainacc;
        test_acc(gain, threshold) = mnist_simple.activations(threshold).thresold(gain).gain.testacc;
        k(gain, threshold) = mnist_simple.activations(threshold).thresold(gain).gain.k;

    end

end

subplot(3, 6, 1 +1);

heatmap(train_acc, 'CellLabelColor','none');
clim([.2 1]);

colormap jet;
colorbar off

Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
Ax.GridVisible = 'off'

subplot(3, 6, 7 + 1);

heatmap(test_acc, 'CellLabelColor','none');
clim([.2 1]);

colormap jet;
colorbar off

Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
Ax.GridVisible = 'off'

subplot(3, 6, 13 + 1);

heatmap(log(k), 'CellLabelColor','none');
clim([2 8]);

colormap jet;
colorbar off

Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
Ax.GridVisible = 'off'

%xlabel('threshold');

train_acc = zeros(11, 11);
test_acc = zeros(11, 11);
k = zeros(11, 11);

for gain = 1:11

    for threshold = 1:11

        train_acc(gain, threshold) = mnist_complex.activations(threshold).thresold(gain).gain.trainacc;
        test_acc(gain, threshold) = mnist_complex.activations(threshold).thresold(gain).gain.testacc;
        k(gain, threshold) = mnist_complex.activations(threshold).thresold(gain).gain.k;

    end

end

subplot(3, 6, 1 + 2);

heatmap(train_acc, 'CellLabelColor','none');
clim([.2 1]);

colormap jet;
colorbar off

Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
Ax.GridVisible = 'off'

subplot(3, 6, 7 + 2);

heatmap(test_acc, 'CellLabelColor','none');
clim([.2 1]);

colormap jet;
colorbar off

Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
Ax.GridVisible = 'off'

subplot(3, 6, 13 + 2);

heatmap(log(k), 'CellLabelColor','none');
clim([2 8]);

colormap jet;
colorbar off

%xlabel('threshold');

Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
Ax.GridVisible = 'off'

train_acc = zeros(11, 11);
test_acc = zeros(11, 11);
k = zeros(11, 11);

for gain = 1:11

    for threshold = 1:11

        train_acc(gain, threshold) = cifar_lgn.activations(threshold).thresold(gain).gain.trainacc;
        test_acc(gain, threshold) = cifar_lgn.activations(threshold).thresold(gain).gain.testacc;
        k(gain, threshold) = cifar_lgn.activations(threshold).thresold(gain).gain.k;

    end

end

subplot(3, 6, 1 + 3);

heatmap(train_acc, 'CellLabelColor','none');
clim([.1 .6]);

colormap jet;
colorbar off

Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
Ax.GridVisible = 'off'

subplot(3, 6, 7 + 3);

heatmap(test_acc, 'CellLabelColor','none');
clim([.1 .35]);

colormap jet;
colorbar off

Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
Ax.GridVisible = 'off'

subplot(3, 6, 13 + 3);

heatmap(log(k), 'CellLabelColor','none');
clim([2 8]);

colormap jet;
colorbar off

Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
Ax.GridVisible = 'off'

%xlabel('threshold');

train_acc = zeros(11, 11);
test_acc = zeros(11, 11);
k = zeros(11, 11);

for gain = 1:11

    for threshold = 1:11

        train_acc(gain, threshold) = cifar_simple.activations(threshold).thresold(gain).gain.trainacc;
        test_acc(gain, threshold) = cifar_simple.activations(threshold).thresold(gain).gain.testacc;
        k(gain, threshold) = cifar_simple.activations(threshold).thresold(gain).gain.k;

    end

end

subplot(3, 6, 1 +4);

heatmap(train_acc, 'CellLabelColor','none');
clim([.1 .6]);

colormap jet;
colorbar off

Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
Ax.GridVisible = 'off'

subplot(3, 6, 7 + 4);

heatmap(test_acc, 'CellLabelColor','none');
clim([.1 .35]);

colormap jet;
colorbar off

Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
Ax.GridVisible = 'off'

subplot(3, 6, 13 + 4);

heatmap(log(k), 'CellLabelColor','none');
clim([2 8]);

colormap jet;
colorbar off

Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
Ax.GridVisible = 'off'

%xlabel('threshold');

train_acc = zeros(11, 11);
test_acc = zeros(11, 11);
k = zeros(11, 11);

for gain = 1:11

    for threshold = 1:11

        train_acc(gain, threshold) = cifar_complex.activations(threshold).thresold(gain).gain.trainacc;
        test_acc(gain, threshold) = cifar_complex.activations(threshold).thresold(gain).gain.testacc;
        k(gain, threshold) = cifar_complex.activations(threshold).thresold(gain).gain.k;

    end

end

subplot(3, 6, 1 + 5);

heatmap(train_acc, 'CellLabelColor','none');
clim([.1 .6]);

colormap jet;
colorbar off

Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
Ax.GridVisible = 'off'

subplot(3, 6, 7 + 5);

heatmap(test_acc, 'CellLabelColor','none');
clim([.1 .35]);

colormap jet;
colorbar off

Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
Ax.GridVisible = 'off'

subplot(3, 6, 13 + 5);

heatmap(log(k), 'CellLabelColor','none');
clim([2 8]);

colormap jet;
colorbar off

Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
Ax.GridVisible = 'off'

%xlabel('threshold');

set(gcf,'color','w');
