
gabor = zeros(64, 64);

x_std = 5;
y_std = 5;
k = .5;
offset = 0;
angle = pi/4;

for x = 1:64

    x_coordinate = x - 32;

     for y = 1:64

         y_coordinate = y - 32;

          gabor(x, y) = 1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);

     end

end

image_gabor = (gabor - min(gabor, [], 'all'));
image_gabor = image_gabor./max(image_gabor, [], 'all');

image_gabor = imresize(image_gabor, 10);

imshow(image_gabor)