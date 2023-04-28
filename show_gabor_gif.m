    
for frame = 1:100

    angle = 0;
    
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

    gabor = zscore(gabor, [], 'all');
    gabor = gabor - gabor(1);


    view = (30);

    h = figure;
    
    surf(gabor);
    
    set(gcf,'color','w');
    
    set(gca,'ZColor','none')
    
    ax = gca;
    ax.FontSize = 14; 
    
    xticks([1 50]);
    yticks([1 50]);
    
    xticklabels({'0', '1'})
    yticklabels({'1', '0'})
    
    colormap(jet);
    grid off;
    box off;

    clim([-3 3]);
    zlim([-6 6]);
    xlim([0 50]);
    ylim([0 50]);
     
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