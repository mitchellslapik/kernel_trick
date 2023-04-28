
svm_on_pixels = load('svm_on_pixels.mat');
svm_on_simple = load('svm_on_simple.mat');
svm_on_complex = load('svm_on_complex.mat');

svm_on_simple_random = load('svm_on_simple_random.mat');
svm_on_complex_random = load('svm_on_complex_random.mat');

svm_on_simple_random_no_rf = load('svm_on_simple_random_no_rf.mat');
svm_on_complex_random_no_rf = load('svm_on_complex_random_no_rf.mat');

train_ces = [svm_on_pixels.train_ces',...
    svm_on_simple.train_ces',...
    svm_on_complex.train_ces',...
    svm_on_simple_random.train_ces',...
    svm_on_complex_random.train_ces',...
    svm_on_simple_random_no_rf.train_ces',...
    svm_on_complex_random_no_rf.train_ces'];


train_accs = [svm_on_pixels.train_accs',...
    svm_on_simple.train_accs',...
    svm_on_complex.train_accs',...
    svm_on_simple_random.train_accs',...
    svm_on_complex_random.train_accs',...
    svm_on_simple_random_no_rf.train_accs',...
    svm_on_complex_random_no_rf.train_accs'];

test_ces = [svm_on_pixels.test_ces',...
    svm_on_simple.test_ces',...
    svm_on_complex.test_ces',...
    svm_on_simple_random.test_ces',...
    svm_on_complex_random.test_ces',...
    svm_on_simple_random_no_rf.test_ces',...
    svm_on_complex_random_no_rf.test_ces'];


test_accs = [svm_on_pixels.test_accs',...
    svm_on_simple.test_accs',...
    svm_on_complex.test_accs',...
    svm_on_simple_random.test_accs',...
    svm_on_complex_random.test_accs',...
    svm_on_simple_random_no_rf.test_accs',...
    svm_on_complex_random_no_rf.test_accs'];

scatter(1:7, train_accs, 50, 'b', 'filled');
alpha(.5);
hold on;

scatter(1:7, test_accs, 50, [0 0 0], 'filled');
alpha(.5);
hold on;

figure;

scatter(1:7, train_ces, 50, 'b', 'filled');
alpha(.5);
hold on;

scatter(1:7, test_ces, 50, [0 0 0], 'filled');
alpha(.5);
hold on;
