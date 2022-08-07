clear;
clc;
%% data all
X = load("./cola_store/X.txt");
y = load("./cola_store/y.txt");
color_vec = 'br';
for label = 0:1
    plot3(X(y == label, 1), X(y == label, 2), X(y == label, 3), sprintf(".%c", color_vec(label + 1)));
    hold on
end
grid on
axis equal
xlabel("B");
ylabel("G");
zlabel("R");

%% load data
color_name_vec = ["Yellow", "Orange", "Green", "White", "Red", "Blue"];
color_rgb_cell = cell(6, 1);
color_hsv_cell = cell(6, 1);
for i = 1:6
    % rgb
    filename = sprintf("./cola_store/color_mat_%s.txt", color_name_vec(i));
    tmp = load(filename);
    color_rgb_cell{i} = tmp;
    % hsv
    filename = sprintf("./cola_store/color_mat_%s_hsv.txt", color_name_vec(i));
    tmp = load(filename);
    color_hsv_cell{i} = tmp;
end


%% bgr, all data
figure(1);
subplot(1, 2, 1)
draw_all_faces(color_rgb_cell, color_rgb_cell, [1, 1, 1]);
xlabel("B")
ylabel("G")
zlabel("R")
grid on
title("BGR Space")
axis equal


%% hsv, all data
subplot(1, 2, 2)
draw_all_faces(color_hsv_cell, color_rgb_cell, [1, 1, 1]);
xlabel("H")
ylabel("S")
zlabel("V")
title("HSV Space")
grid on
axis equal 

%% hsv, white and yellow
scale = [1, 1, 1];
draw_point_with_color2(color_hsv_cell{1}, 'y', scale);
hold on
draw_point_with_color2(color_hsv_cell{4}, 'r', scale);
xlabel("H")
ylabel("S")
zlabel("V")
axis equal

%% find cube
border_cell = cell(6, 1);
for color_label = 1:6  % "Yellow", "Orange", "Green", "White", "Red", "Blue"
%     color_label = 1;  
    border_mat = zeros(3, 2);  % hsv max min
    cube_mat = zeros(8, 3);
    for i = 1:3
        data = color_hsv_cell{color_label};
        [~, idx_min] = min(data(:, i));
        [~, idx_max] = max(data(:, i));
    %     disp(data(idx_min, :))
        border_mat(i, 1) = data(idx_min, i);
        border_mat(i, 2) = data(idx_max, i);
    end
%     point1 = [border_mat(1, 1), border_mat(2, 1), border_mat(3, 1)];
%     point2 = [border_mat(1, 2), border_mat(2, 2), border_mat(3, 2)]
%     plotcube(point1 - point2, point2, .8)
    border_cell{color_label} = border_mat;
end




%% To dataset, rgb
target_idx = [1, 3, 4, 6];
X = [];
y = [];
for i = 1:length(target_idx)
    X = [X; color_rgb_cell{i}];
    tmp_y = ones(length(color_rgb_cell{i}) , 1) * i;
    y = [y; tmp_y];
end

%% To dataset, hsv
target_idx = [1, 3, 4, 6];
X = [];
y = [];
for i = 1:length(target_idx)
    X = [X; color_hsv_cell{i}];
    tmp_y = ones(length(color_hsv_cell{i}) , 1) * i;
    y = [y; tmp_y];
end

%% dataset, red & orange | other. rgb or hsv
idx_ro = y == 2 | y == 5;
y(idx_ro, :) = 1;
y(~idx_ro) = 0;

%% 
plot3(X(idx_ro, 1), X(idx_ro, 2), X(idx_ro, 3), '.r');
hold on
plot3(X(~idx_ro, 1), X(~idx_ro, 2), X(~idx_ro, 3), '.g');
xlabel("B")
ylabel("G")
zlabel("R")
title("RGB Space")
legend("红色或橙色", '其他颜色')
grid off
box on

%% hsv, orange red
% scale = [0.01, 0.1, 0.2];
scale = [1, 1, 1];
draw_point_with_color2(color_hsv_cell{2}, 'g', scale);
hold on
draw_point_with_color2(color_hsv_cell{5}, 'r', scale);
xlabel("H")
ylabel("S")
zlabel("V")
axis equal

%% rgb, orange red
% scale = [0.01, 0.1, 0.2];
scale = [1, 1, 1];
draw_point_with_color2(color_rgb_cell{2}, '.', scale);
hold on
draw_point_with_color2(color_rgb_cell{5}, 'r', scale);
xlabel("B")
ylabel("G")
zlabel("R")
legend("橙色", '红色')
title("橙色和红色 (RGB空间)")
box on

%% my function
function [] = draw_all_faces(color_cell, color_rgb_cell, scale)
    for idx_face = 1:6
        color_mat = color_rgb_cell{idx_face} / 255.0;
        draw_point_with_color(color_cell{idx_face}, color_mat(:, [3, 2, 1]), scale);
        hold on
    end
end

function [] = draw_point_with_color(color_vec, color_rgb,scale)
    % color_vec: shape = (N, 3)
    N = length(color_vec);
    for i = 1:N
        x = color_vec(i, 1) * scale(1);
        y = color_vec(i, 2) * scale(2);
        z = color_vec(i, 3) * scale(3);
        plot3(x, y, z, '.', 'MarkerSize', 10, color=color_rgb(i, :));
        hold on
    end
end

function [] = draw_point_with_color2(color_vec, color, scale)
    % color_vec: shape = (N, 3)
    N = length(color_vec);
    for i = 1:N
%         x = cos(color_vec(i, 1) * 2 * pi) * scale(1);
        x = color_vec(i, 1) * scale(1);
        y = color_vec(i, 2) * scale(2);
        z = color_vec(i, 3) * scale(3);
        f = sprintf(".%s", color);
        plot3(x, y, z, f, 'MarkerSize', 10);
        hold on
    end
end
