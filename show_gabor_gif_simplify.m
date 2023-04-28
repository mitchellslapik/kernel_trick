    
for frame = 1:100

    angle = 6;
    
    gabor = zeros(50, 50);
    
    x_std = 7.5;
    y_std = 7.5;
    k = .30;
    offset = frame /100 * 2 *pi;
    angle = angle*pi/8;
    
    for x = 1:50
    
        x_coordinate = x - 25;
    
        for y = 1:50
    
            y_coordinate = y - 25;
    
            gabor(x, y) = 1/(2*pi*x_std*y_std)*exp(-x_coordinate^2/(2*x_std^2) - y_coordinate^2/(2*y_std^2))*cos(k*(x_coordinate*cos(angle) + y_coordinate*sin(angle)) - offset);
    
        end
    
    end

    gabor = gabor / .005;
    gabor = gabor + .5;

    h = figure;

    gabor = imresize(gabor, 10);
    
    imshow(gabor);
    
    set(gcf,'color','w');
    
    %set(gca,'ZColor','none')
   
    pic = getframe(h);
    im = frame2im(pic);
    [imind, cm] = rgb2ind(im, 256);

    if frame == 1

        imwrite(imind, cm, "complex.gif", 'gif', 'DelayTime', 0.025, 'Loopcount', inf); 

    else

        imwrite(imind, cm, "complex.gif", 'gif', 'DelayTime', 0.025, 'WriteMode', 'append'); 

    end 
    
    close;

end