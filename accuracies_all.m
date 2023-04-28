
Set1 = brewermap(6,"Set1");

cifar_svm_on_pixels = load('cifar simple complex/svm_on_pixels.mat');
cifar_svm_on_simple = load('cifar simple complex/svm_on_simple.mat');
cifar_svm_on_complex = load('cifar simple complex/svm_on_complex.mat');

cifar_svm_on_simple_random = load('cifar simple complex/svm_on_simple_random.mat');
cifar_svm_on_complex_random = load('cifar simple complex/svm_on_complex_random.mat');

cifar_svm_on_simple_random_no_rf = load('cifar simple complex/svm_on_simple_random_no_rf.mat');
cifar_svm_on_complex_random_no_rf = load('cifar simple complex/svm_on_complex_random_no_rf.mat');

mnist_svm_on_pixels = load('mnist simple complex/svm_on_pixels.mat');
mnist_svm_on_simple = load('mnist simple complex/svm_on_simple.mat');
mnist_svm_on_complex = load('mnist simple complex/svm_on_complex.mat');

mnist_svm_on_simple_random = load('mnist simple complex/svm_on_simple_random.mat');
mnist_svm_on_complex_random = load('mnist simple complex/svm_on_complex_random.mat');

mnist_svm_on_simple_random_no_rf = load('mnist simple complex/svm_on_simple_random_no_rf.mat');
mnist_svm_on_complex_random_no_rf = load('mnist simple complex/svm_on_complex_random_no_rf.mat');
% 
% h = figure('Units','pixels', 'Position', [300, 300, 800, 400]);
% 
% subplot(1, 2, 1);
% 
% scatter(mnist_svm_on_pixels.train_accs, mnist_svm_on_pixels.test_accs, 100, [0 0 0], 'filled');
% hold on; 
% 
% scatter(mnist_svm_on_simple.train_accs, mnist_svm_on_simple.test_accs, 100, Set1(2, :), 'filled');
% hold on; 
% 
% scatter(mnist_svm_on_complex.train_accs, mnist_svm_on_complex.test_accs, 100, Set1(1, :), 'filled');
% hold on; 
% 
% alpha(.5);
% 
% ax = gca;
% ax.FontSize = 14; 
% 
% subplot(1, 2, 2);
% 
% scatter(cifar_svm_on_pixels.train_accs, cifar_svm_on_pixels.test_accs, 100, [0 0 0], 'filled');
% hold on; 
% 
% scatter(cifar_svm_on_simple.train_accs, cifar_svm_on_simple.test_accs, 100, Set1(2, :), 'filled');
% hold on; 
% 
% scatter(cifar_svm_on_complex.train_accs, cifar_svm_on_complex.test_accs, 100, Set1(1, :), 'filled');
% hold on; 
% 
% alpha(.5);
% 
% set(gcf,'color','w');
% 
% ax = gca;
% ax.FontSize = 14; 

% 
% h = figure('Units','pixels', 'Position', [300, 300, 800, 400]);
% 
% subplot(1, 2, 1);
% 
% scatter(mnist_svm_on_pixels.train_ces, mnist_svm_on_pixels.test_ces, 100, [0 0 0], 'filled');
% hold on; 
% 
% scatter(mnist_svm_on_simple.train_ces, mnist_svm_on_simple.test_ces, 100, Set1(2, :), 'filled');
% hold on; 
% 
% scatter(mnist_svm_on_complex.train_ces, mnist_svm_on_complex.test_ces, 100, Set1(1, :), 'filled');
% hold on; 
% 
% alpha(.5);
% 
% ax = gca;
% ax.FontSize = 14; 
% 
% subplot(1, 2, 2);
% 
% scatter(cifar_svm_on_pixels.train_ces, cifar_svm_on_pixels.test_ces, 100, [0 0 0], 'filled');
% hold on; 
% 
% scatter(cifar_svm_on_simple.train_ces, cifar_svm_on_simple.test_ces, 100, Set1(2, :), 'filled');
% hold on; 
% 
% scatter(cifar_svm_on_complex.train_ces, cifar_svm_on_complex.test_ces, 100, Set1(1, :), 'filled');
% hold on; 
% 
% alpha(.5);
% 
% set(gcf,'color','w');
% 
% ax = gca;
% ax.FontSize = 14; 
% 


h = figure('Units','pixels', 'Position', [300, 300, 600, 600]);

subplot(3, 3, 2);

scatter(mnist_svm_on_pixels.train_accs*100, mnist_svm_on_pixels.test_accs*100, 100, [0 0 0], 'filled');
hold on; 

scatter(mnist_svm_on_simple.train_accs*100, mnist_svm_on_simple.test_accs*100, 100, Set1(2, :), 'filled');
hold on; 

scatter(mnist_svm_on_complex.train_accs*100, mnist_svm_on_complex.test_accs*100, 100, Set1(1, :), 'filled');
hold on; 

ylim([85 100]);
xlim([85 100]);

title('mnist', 'FontSize', 14', 'FontWeight', 'normal');

ylabel('test accuracy (%)', 'FontSize', 14);

xticks([]);

plot([85 100], [85 100], 'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'LineStyle', '--');
hold on;

alpha(.5);

ax = gca;
ax.FontSize = 14; 

subplot(3, 3, 3);

scatter(cifar_svm_on_pixels.train_accs*100, cifar_svm_on_pixels.test_accs*100, 100, [0 0 0], 'filled');
hold on; 

scatter(cifar_svm_on_simple.train_accs*100, cifar_svm_on_simple.test_accs*100, 100, Set1(2, :), 'filled');
hold on; 

scatter(cifar_svm_on_complex.train_accs*100, cifar_svm_on_complex.test_accs*100, 100, Set1(1, :), 'filled');
hold on; 

alpha(.5);

title('cifar10', 'FontSize', 14', 'FontWeight', 'normal');

set(gcf,'color','w');

ylim([20 50]);
xlim([20 50]);

xticks([]);

plot([20 50], [20 50], 'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'LineStyle', '--');
hold on;

ax = gca;
ax.FontSize = 14; 

subplot(3, 3, 5);

scatter(mnist_svm_on_pixels.train_accs*100, mnist_svm_on_pixels.test_accs*100, 100, [0 0 0], 'filled');
hold on; 

scatter(mnist_svm_on_simple_random.train_accs*100, mnist_svm_on_simple_random.test_accs*100, 100, Set1(2, :), 'filled');
hold on; 

scatter(mnist_svm_on_complex_random.train_accs*100, mnist_svm_on_complex_random.test_accs*100, 100, Set1(1, :), 'filled');
hold on; 

ylim([85 100]);
xlim([85 100]);

ylabel('test accuracy (%)', 'FontSize', 14);

xticks([]);

plot([85 100], [85 100], 'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'LineStyle', '--');
hold on;

alpha(.5);

ax = gca;
ax.FontSize = 14; 

subplot(3, 3, 6);

scatter(cifar_svm_on_pixels.train_accs*100, cifar_svm_on_pixels.test_accs*100, 100, [0 0 0], 'filled');
hold on; 

scatter(cifar_svm_on_simple_random.train_accs*100, cifar_svm_on_simple_random.test_accs*100, 100, Set1(2, :), 'filled');
hold on; 

scatter(cifar_svm_on_complex_random.train_accs*100, cifar_svm_on_complex_random.test_accs*100, 100, Set1(1, :), 'filled');
hold on; 

ylim([20 50]);
xlim([20 50]);

xticks([]);

plot([20 50], [20 50], 'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'LineStyle', '--');
hold on;

alpha(.5);

set(gcf,'color','w');

ax = gca;
ax.FontSize = 14; 

subplot(3, 3, 8);

scatter(mnist_svm_on_pixels.train_accs*100, mnist_svm_on_pixels.test_accs*100, 100, [0 0 0], 'filled');
hold on; 

scatter(mnist_svm_on_simple_random_no_rf.train_accs*100, mnist_svm_on_simple_random_no_rf.test_accs*100, 100, Set1(2, :), 'filled');
hold on; 

scatter(mnist_svm_on_complex_random_no_rf.train_accs*100, mnist_svm_on_complex_random_no_rf.test_accs*100, 100, Set1(1, :), 'filled');
hold on; 

ylim([85 100]);
xlim([85 100]);

ylabel('test accuracy (%)', 'FontSize', 14);
xlabel('train accuracy (%)', 'FontSize', 14);

plot([85 100], [85 100], 'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'LineStyle', '--');
hold on;

alpha(.5);

ax = gca;
ax.FontSize = 14; 

subplot(3, 3, 9);

scatter(cifar_svm_on_pixels.train_accs*100, cifar_svm_on_pixels.test_accs*100, 100, [0 0 0], 'filled');
hold on; 

scatter(cifar_svm_on_simple_random_no_rf.train_accs*100, cifar_svm_on_simple_random_no_rf.test_accs*100, 100, Set1(2, :), 'filled');
hold on; 

scatter(cifar_svm_on_complex_random_no_rf.train_accs*100, cifar_svm_on_complex_random_no_rf.test_accs*100, 100, Set1(1, :), 'filled');
hold on; 

ylim([20 50]);
xlim([20 50]);

plot([20 50], [20 50], 'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'LineStyle', '--');
hold on;

alpha(.5);

xlabel('train accuracy (%)', 'FontSize', 14);

set(gcf,'color','w');

ax = gca;
ax.FontSize = 14; 

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
