gabor = zeros(64, 64);

x_std = 5;
y_std = 5;
k = .5;
offset = 0;
angle = 0*pi/8;

for x = 1:64

    x_coordinate = x - 32;

     for y = 1:64

         y_coordinate = y - 32;

          gabor(x, y) = 1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);

     end

end

gabor = (gabor - min(gabor, [], 'all')) / (max(gabor, [], 'all') - min(gabor, [], 'all'));

gabor = imresize(gabor, 10);

save('real_gabor', 'gabor');

gabor = zeros(64, 64);

x_std = 5;
y_std = 5;
k = .5;
offset = 0;
angle = 0*pi/8;

for x = 1:64

    x_coordinate = x - 32;

     for y = 1:64

         y_coordinate = y - 32;

          gabor(x, y) = 1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*randn();%cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);

     end

end

gabor = (gabor - min(gabor, [], 'all')) / (max(gabor, [], 'all') - min(gabor, [], 'all'));

gabor = imresize(gabor, 10);

save('random_gabor', 'gabor');

gabor = zeros(64, 64);

x_std = 5;
y_std = 5;
k = .5;
offset = 0;
angle = 0*pi/8;

for x = 1:64

    x_coordinate = x - 32;

     for y = 1:64

         y_coordinate = y - 32;

          gabor(x, y) = randn(); %1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*randn();%cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);

     end

end

gabor = (gabor - min(gabor, [], 'all')) / (max(gabor, [], 'all') - min(gabor, [], 'all'));

gabor = imresize(gabor, 10);

save('random', 'gabor');