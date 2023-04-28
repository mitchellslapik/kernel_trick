

data = -5:.1:5;

for threshold = 1:11

    for gain = 1:11

        subplot(11, 11, (gain - 1)*11 + threshold)

        thresholded = data - (threshold - 1)/2.5 + 2;

        thresholded(thresholded < 0) = 0;

        gained = thresholded * 10^((gain - 1)/2.5 - 2);
                
        final = 1./(1 + exp(-gained));
        
        plot(data, final, 'LineWidth', 2);
        ylim([0.5 1]);

        axis off

        box off;
        set(gcf,'color','w');

    end

end

%%

figure;



gained = -5:.1:5;
        
final = 1./(1 + exp(-gained));

plot(-5:.1:5, final, 'LineWidth', 3);
ylim([0 1]);

xlabel('contrast', 'FontSize', 14)

ylabel('neural response', 'FontSize', 14)

ax = gca;
ax.FontSize = 14;

box off;
set(gcf,'color','w');


%%


figure;

data = -5:.1:5;

gain = 5;
threshold = 2;

thresholded = data - (threshold - 1)/2.5 + 2;

thresholded(thresholded < 0) = 0;

gained = thresholded * 10^((gain - 1)/2.5 - 2);
        
final = 1./(1 + exp(-gained));

plot(-5:.1:5, final, 'LineWidth', 3);
axis off;

box off;
set(gcf,'color','w');