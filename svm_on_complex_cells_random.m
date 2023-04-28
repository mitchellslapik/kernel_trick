directory = dir('data/');

train_accs = [];
train_ces = [];

test_accs = [];
test_ces = [];

activations = struct();

directory = directory(4:end);

for train_index = 1:6

    for test_index = 1:5

        possible_test_indexes = 1:6;

        possible_test_indexes(train_index) = [];

        corrected_test_index = possible_test_indexes(test_index);
    
        train = load(['data/' directory(train_index).name]);
        test = load(['data/' directory(corrected_test_index).name]);

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
        
        for angle = 1:8
            
            gabor = zeros(64, 64);
            
            x_std = 5;
            y_std = 5;
            k = .5;
            offset = 0;
            angle = angle*pi/8;
            
            for x = 1:64
            
                x_coordinate = x - 32;
            
                 for y = 1:64
            
                     y_coordinate = y - 32;
            
                      gabor(x, y) = 1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*randn();%*cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);
            
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
            
                      gabor(x, y) = 1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*randn();%*cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);
            
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
        
        nb_classes = 10;
        layers = [ ...
        imageInputLayer([32 32 8])
        fullyConnectedLayer(nb_classes)
        softmaxLayer
        classificationLayer];
        
        options = trainingOptions('adam', ...
        'Shuffle','every-epoch', ...
        'MaxEpochs', 5, ...
        'MiniBatchSize', 100, ...
        'ValidationData',{XTest_gray, YTest}, ...
        'ValidationFrequency', 10, ...
        'Plots','training-progress');
        
        net = trainNetwork(XTrain_gray, YTrain, layers, options);
        
        YPred = predict(net,XTrain_gray);
        acc = mean_accuracy( YTrain, YPred );
        ce = mean_cross_entropy( YTrain, YPred );
        
        train_accs = [train_accs acc];
        train_ces = [train_ces ce];
        
        YPred = predict(net,XTest_gray);
        acc = mean_accuracy( YTest, YPred );
        ce = mean_cross_entropy( YTest, YPred );

        test_accs = [test_accs acc];
        test_ces = [test_ces ce];

        activations(train_index).activation = XTrain_gray;

        delete(findall(0));

    end

end

save('svm_on_complex_random.mat', 'train_accs', 'train_ces', 'test_accs', 'test_ces', 'activations');
