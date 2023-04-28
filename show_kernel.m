
Set1 = brewermap(6,"Set1");

x = randn(200, 1);
y = randn(200, 1);
z = x.^2 + y.^2;
z = 1./(5 + exp(z));

h = figure('Units','pixels', 'Position', [300, 300, 900, 400]);

subplot(1, 2, 1);

red = z > .125;
blue = z < .075;

scatter(x(red), y(red), 100,  Set1(1, :), 'filled');
alpha(.5);
hold on;

scatter(x(blue), y(blue), 100,  Set1(2, :), 'filled');
alpha(.5);

%title('input data', 'FontSize', 14', 'FontWeight', 'normal');

ax = gca;
ax.FontSize = 18; 

xlim([-3 3]);
ylim([-3 3]);

xlabel('x', 'FontSize', 18);
ylabel('y', 'FontSize', 18);

grid on;

box off;

axis square;

subplot(1, 2, 2);

scatter3(x(red), y(red), z(red), 100,  Set1(1, :), 'filled')
alpha(.5);
hold on;

scatter3(x(blue), y(blue), z(blue), 100,  Set1(2, :), 'filled')
alpha(.5);

[X,Y] = meshgrid(-4:.4:4);
Z = (X.^2 + Y.^2);
%Z = 1./(3 + Z);
Z = 1./(5 + exp(Z));

mesh(X,Y,Z, 'EdgeColor', [.5 .5 .5],'FaceColor', 'none');

set(gcf,'color','w');

ax = gca;
ax.FontSize = 18; 

zticks([.1 .2]);
zticklabels({'.1', '.2'});

grid off;

%title('kernel trick', 'FontSize', 14, 'FontWeight', 'normal');

xlim([-3 3]);
ylim([-3 3]);
%zlim([0 2]);

[X,Y] = meshgrid(-4:1:4);

h = mesh(X,Y,.1*ones(9, 9), 'FaceColor', [.5 .5 .5], 'EdgeColor', 'none', 'LineWidth', 3);
alpha(h, .2);
