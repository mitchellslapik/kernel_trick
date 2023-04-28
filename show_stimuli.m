
cifar = load('cifar simple complex/data/data_batch_1.mat');

mnist = load('mnist simple complex/data/mnist.mat');

mnist_X_Train = mnist.XTrain;

mnist_Y_Train = mnist.YTrain;

cifar_X_Train = cifar.data;

cifar_Y_Train = cifar.labels;

cifar_X_Train = reshape(cifar_X_Train, [10000, 32, 32, 3]);

h = figure('Units','pixels', 'Position', [300, 300, 800, 400]);
ha = tight_subplot(10, 20,[0.01 0.01],[0.1 .1],[0.05 0.05]);

for row = 1:10
    
    for n = 1:10
    
        mnist_indexes = find(mnist_Y_Train == n - 1);
    
        cifar_indexes = find(cifar_Y_Train == n - 1);

        mnist_index = mnist_indexes(row);
    
        cifar_index = cifar_indexes(row);

        axes(ha((row - 1)*20 + n));
    
        picture = mnist_X_Train(:, :, :, mnist_index);

        picture = squeeze(picture);

        imshow(picture);
    
        axis equal;

        axes(ha((row - 1)*20 + n + 10));
    
        picture = cifar_X_Train(cifar_index, :, :, :);

        picture = squeeze(picture);

        picture = permute(picture, [2 1 3]);

        imshow(picture);
    
        axis equal;

    end

end

for n = 1:10

    axes(ha(n));

    title(num2str(n - 1), 'FontSize', 14, 'FontWeight', 'normal');
    

end

labels = {'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'};

for n = 1:10

    axes(ha(n + 10));

    title(labels{n}, 'FontSize', 14, 'FontWeight', 'normal');
    

end

set(gcf,'color','w');