function visualize_pose(image, joint, visible)
% visualize pose data
% IM        is the original image
% JOINT     is the [2 * 14] matrix where the 1st row denotes X cordinate
%           and 2nd row denotes Y cordinate
% VISIBLE   is the visibility of joints
%
% platero.yang@gmail.com
% 8 Sept 2014

x           = joint(1, :);
y           = joint(2, :);

% visulize
% NOTE: COORDINATES OF INVISIBLE JOINTS ARE TRIVIAL
f = figure; colors = colormap(lines(10));
imshow(image); hold on;

% Right leg
if visible(1) && visible(2), plot(x(1:2), y(1:2), '-', 'Color', colors(1, :), 'LineWidth', 2); end
if visible(2) && visible(3), plot(x(2:3), y(2:3), '-', 'Color', colors(2, :), 'LineWidth', 2); end

% Left leg
if visible(4) && visible(5), plot(x(4:5), y(4:5), '--', 'Color', colors(3, :), 'LineWidth', 2); end
if visible(5) && visible(6), plot(x(5:6), y(5:6), '--', 'Color', colors(4, :), 'LineWidth', 2); end

% Right arm
if visible(7) && visible(8), plot(x(7:8), y(7:8), '-', 'Color', colors(5, :), 'LineWidth', 2); end
if visible(8) && visible(9), plot(x(8:9), y(8:9), '-', 'Color', colors(6, :), 'LineWidth', 2); end

% Left leg
if visible(10) && visible(11), plot(x(10:11), y(10:11), '--', 'Color', colors(7, :), 'LineWidth', 2); end
if visible(11) && visible(12), plot(x(11:12), y(11:12), '--', 'Color', colors(8, :), 'LineWidth', 2); end

% Hip to neck
if visible(3) && visible(4) && visible(13)
    plot([(x(3)+x(4))/2, x(13)], [(y(3)+y(4))/2, y(13)], '-', 'Color', colors(9, :), 'LineWidth', 2); 
end

% Neck to head
if visible(13) && visible(14), plot(x(13:14), y(13:14), '-', 'Color', colors(10, :), 'LineWidth', 2); end

hold on; pause; close(f);
end