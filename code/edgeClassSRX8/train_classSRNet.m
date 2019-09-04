function [net, info] = train_classSRNet(imdb,varargin)
%cnn_image_depth
%%  ×Ô¶¨Òå

opts.train = struct() ;
opts.train.derOutputs = {'edgeloss', 1,'SRloss',1} ;
opts.train.batchSize=80;
opts.train.numSubBatches = 4 ;
opts.train.gpus=[4];
opts.train.weightClipping=true;
% opts.train.weightClipping=false;
opts.train.theta=0.005;
opts.train.numEpochs = 350 ;
% opts.train.learningRate= 0.01*[logspace(-1,-2,20) logspace(-2,-3,20) logspace(-3,-4,20) logspace(-4,-5,20) logspace(-5,-6,20)...
% logspace(-6,-7,20) logspace(-6,-7,20) logspace(-6,-7,20) logspace(-6,-7,20) ...
%      logspace(-6,-7,20) logspace(-6,-7,20) logspace(-6,-7,20) logspace(-6,-7,20) logspace(-6,-7,20) logspace(-6,-7,20) ];
opts.train.learningRate=1e-2*[1e-3*ones(1,10) ...
    5e-4*ones(1,5) 1e-4*ones(1,5) ...
    5e-5*ones(1,10) 1e-5*ones(1,10)...
    5e-6*ones(1,10) 1e-6*ones(1,10) 5e-7*ones(1,10)...
    1e-7*ones(1,10) 5e-8*ones(1,10) 1e-8*ones(1,10)...
    5e-9*ones(1,10) 1e-9*ones(1,10) 5e-10*ones(1,10)...
    ] ;
% opts.train.learningRate=0;
%%
opts.dataDir = fullfile('data','model');

opts.expDir = fullfile('data', 'model') ;


opts.modelType = 'classSRNet';
opts.network = [] ;

opts.networkType = 'dagnn' ;
opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

% sfx = opts.modelType ;
% if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end  %%depth_resnet-bnorm
% sfx = [sfx '-' opts.networkType] ;  %%depth_resnet-bnorm-dagnn
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 6 ;
opts.lite = false ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

if isempty(opts.network)
    switch opts.modelType
        case 'classSRNet'
			net = getNet();
            opts.networkType = 'dagnn' ;
            
        otherwise
            %       net = cnn_imagenet_init('model', opts.modelType, ...
            %                               'batchNormalization', opts.batchNormalization, ...
            %                               'weightInitMethod', opts.weightInitMethod, ...
            %                               'networkType', opts.networkType, ...
            %                               'averageImage', rgbMean, ...
            %                               'colorDeviation', rgbDeviation, ...
            %                               'classNames', imdb.classes.name, ...
            %                               'classDescriptions', imdb.classes.description) ;
    end
else
    net = opts.network ;
    opts.network = [] ;
end

% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

switch opts.networkType
    %   case 'simplenn', trainFn = @cnn_train ;
    case 'dagnn', trainFn = @cnn_train_dag ;
end

[net, info] = trainFn(net, imdb, getBatchFn(opts, net.meta), ...
    'expDir', opts.expDir, ...
    opts.train) ;

% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------

% net = cnn_depth_resnet_deploy(net) ;
modelPath = fullfile(opts.expDir, 'net-deployed.mat')

switch opts.networkType
    case 'simplenn'
        save(modelPath, '-struct', 'net') ;
    case 'dagnn'
        net_ = net.saveobj() ;
        save(modelPath, '-struct', 'net_') ;
        clear net_ ;
end

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------

% if numel(meta.normalization.averageImage) == 3
%   mu = double(meta.normalization.averageImage(:)) ;
% else
%   mu = imresize(single(meta.normalization.averageImage), ...
%                 meta.normalization.imageSize(1:2)) ;
% end
%
useGpu = numel(opts.train.gpus) > 0 ;
%
% bopts.test = struct(...
%   'useGpu', useGpu, ...
%   'numThreads', opts.numFetchThreads, ...
%   'imageSize',  meta.normalization.imageSize(1:2), ...
%   'cropSize', meta.normalization.cropSize, ...
%   'subtractAverage', mu) ;

%
% % Copy the parameters for data augmentation
% bopts.train = bopts.test ;
% for f = fieldnames(meta.augmentation)'
%   f = char(f) ;
%   bopts.train.(f) = meta.augmentation.(f) ;
% end

fn = @(x,y) getBatch(useGpu,lower(opts.networkType),x,y) ;

% -------------------------------------------------------------------------
function varargout = getBatch(useGpu, networkType, imdb, batch)
% -------------------------------------------------------------------------
if nargout > 0
    grays(:,:,1,:) =single( imdb.images.grays(:,:,1,batch) );
    depths(:,:,1,:)=single(imdb.images.depths(:,:,1,batch));
% here use '-' because wanted label(pre)=depth-label 
% while dataset is label=label-depth
    labels(:,:,1,:)=single(0-imdb.images.labels(:,:,1,batch));
    
    masks(:,:,1,:)=single(imdb.images.masks(:,:,1,batch));
    edges(:,:,1,:)=single(imdb.images.edges(:,:,1,batch));
    
    instance_weights=(masks*1+1);
    randnumber = rand;
    if 0.45 < randnumber  && randnumber < 0.6
        grays=fliplr(grays);
		depths=fliplr(depths);
		edges=fliplr(edges);
		masks=fliplr(masks);
		labels=fliplr(labels);
    end
    if 0.3 < randnumber  && randnumber < 0.45
        grays=rot90(grays,2); 
		depths=rot90(depths,2);
		edges=rot90(edges,2);
		masks=rot90(masks,2);
		labels=rot90(labels,2);
    end
    if 0.15 < randnumber  && randnumber < 0.3
        grays=flipud(grays); 
		depths=flipud(depths);
		edges=flipud(edges);
		masks=flipud(masks);
		labels=flipud(labels);
    end
    if 0 < randnumber  && randnumber < 0.15
        grays=rot90(grays,3);
		depths=rot90(depths,3);
		edges=rot90(edges,3);
		masks=rot90(masks,3);
		labels=rot90(labels,3);
    end
    if useGpu
        grays=gpuArray(grays);
        depths=gpuArray(depths);
        edges=gpuArray(edges);
        masks=gpuArray(masks);
        labels=gpuArray(labels);
        instance_weights=gpuArray(instance_weights);
    end
    switch networkType
        case 'simplenn'
            %       varargout = {data, labels} ;
        case 'dagnn'
            varargout{1} = {...
                'input_d',depths,...
                'input_g', grays,...
                'input_elabel',edges, ...
                'input_dlabel',labels, ...
				'instance_weights',instance_weights...
                } ;
    end
end

