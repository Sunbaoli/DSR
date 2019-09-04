run('~/src/addpath_matconvnet.m')

opts.modelType = 'classSRNet' ;
opts.expDir = fullfile('data', 'model') ;
% imdbPath = '../data/midd_591280_160_0_hole_size_41_stride_10_threshold_0_residue_08-Mar-2018.mat';
% imdbPath = '../data/weightedtoclass_midd_50560_160_0_hole_size_81_stride_20_threshold_0_14-Mar-2018.mat';
imdbPath = '../data/mask2vs1X16classSR_midd_121920_160_0_hole_size_41_stride_10_threshold_0_19-Apr-2018.mat';
imdbPath = '../data/mask2vs1X16classSR_midd_47600_160_0_hole_size_81_stride_20_threshold_0_28-Apr-2018.mat';

if ~exist('imdb')

    imdb=load(imdbPath);
	% imdb.iamges.labels=imdb.images.labels-imdb

end

train_classSRNet(imdb,opts);