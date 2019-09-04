function [depth,max_d,min_d]=trans_depth(depth)
% depth->0.1-0.9;
% y=0.8*x/(max-min)+(0.1max-0.9min)/(max-min)
%  =[0.9(x-min)+0.1(max-x)]/(max-min)
max_d=max(depth(:));
min_d=min(depth(:));
depth=(0.9*(depth-min_d)+0.1*(max_d-depth))/(max_d-min_d);