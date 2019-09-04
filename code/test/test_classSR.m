addpath('~/src/myMfile')
run('~/src/addpath_matconvnet.m')
folder='../test_data';
filepaths = dir(fullfile(folder, '*.mat'));
color_high_fre=true;
%scale=16;
%scale=8; 
scale=4; 
%scale=2;
%method='bilinear';
%method='bicubic';
method='nearest';
gpu=[2]; 

net_path=sprintf('../x%d',scale);

% result_path=sprintf('result/NYU');
mkdir(result_path);
useGpu=true;
for epoch=106%[0 1 10 20 30:10:300] %[110 120 130 140]%[10 20 50 60 70 80]%300:5:350%[1 50 100 150 200 250:10:350]%
    if epoch
    %load(fullfile(net_path, sprintf('net-epoch-%d.mat', epoch)),'net');
    load(fullfile(net_path, 'net-init.mat'));
	else
    load(fullfile(sprintf('../noisyX%d/',scale), 'net-init.mat'),'net' );
	end
    net = dagnn.DagNN.loadobj(net) ;
    net.removeLayer({'edgeloss'});
	net.removeLayer({'SRloss'});
    idx1 = net.getVarIndex('x13');
    net.vars(idx1).precious=true;
    idx2 = net.getVarIndex('softmax') ;
    net.vars(idx2).precious=true;
    
    net.conserveMemory=true;
    net.mode='test';
    gpuDevice(gpu); 
    if useGpu
        net.move('gpu'); 
    end
    % rmse_cm=[];rmse_origin=[];
    % out = net.getVarIndex('predict_conv') ;
    dag = true ;
    for i = 1 : length(filepaths)
	% % % % % % % % % % % % % % % % % % % % % % % % % % result path %%%%%%%%%%%%%%%%%%%%%%
        result_path=sprintf('result/classSR-noise/X%d/%s',scale,filepaths(i).name(1:end-4));
        % result_path=sprintf('result/classSR-noise/X%d-ye/%s',scale,filepaths(i).name(1:end-4));
		if~exist(result_path)
    mkdir(result_path);
	end
    load(fullfile(folder,filepaths(i).name));
        
        im_ycbcr=rgb2ycbcr(im2double(modcrop(color_img,scale)));
    im_gray=im_ycbcr(:,:,1);
    im_gray=normalize_cleanIm(im_gray);
    h=ones(3)/9;
    im_gray=normalize_cleanIm(im_gray-imfilter(im_gray, h, 'symmetric'));
	
    im_label=modcrop(outDepth,scale);
    
	
	edge_=edge(im_label,'canny',0.08);
	
	
	% [im_label,max_d,min_d]=normalize_cleanIm(im_label);
	
	% -------origin------
	% im_depth=imnoise(imresize(im_label,1/scale,method),'gaussian',0,25/255/255);noisy_depth=im_depth;
    im_depth=imresize(im_label,1/scale,method);
    noisy_depth=im_depth;
	[im_depth,max_d,min_d]=normalize_cleanIm(im_depth);
	
	im_depth=imresize(im_depth,scale);
	
	
	input_mask=im_depth*0;
	input_mask(scale/2:scale:end,scale/2:scale:end)=1;
    
    im_edge=edge_+1;
    % im_edge(:,:,2)=edge_*1+1;
        
        input_depth=single(im_depth);
        input_gray=single(im_gray);
        % input_edge=single(im_edge);
        input_mask=single(input_mask);
		
        % input_edgedepth=input_depth;
        % input_edgedepth(:,:,2)=input_edge;
        % instance_weights=single(ones(size(im_edge)));
        if useGpu
            input_depth=gpuArray(input_depth);
            input_gray=gpuArray(input_gray);
            input_mask=gpuArray(input_mask);
            % input_edge=gpuArray(input_edge);
           
        end
        
        
        % run the CNN
        if dag
            net.eval({'input_d',input_depth,...
				'input_g',input_gray,...
				...'input_m',input_mask,...
                ... 'instance_weights',instance_weights...
				}) ;
            pre1=net.vars(idx1).value;
			softmax_pre=net.vars(idx2).value;
            if useGpu
                pre1=gather(pre1);
                input_depth=gather(input_depth);
                softmax_pre=gather(softmax_pre);
                
            end
			% result=inverse_depth(input_depth+pre1,max_d,min_d);
			
			result=inverse_depth(input_depth-pre1,max_d,min_d);
			% result=(input_depth-pre1);
			% softmax_pre=vl_nnsoftmax(pre1);
			% result=getFromEdgeClass(softmax_pre,2);
			% pree = pre1; 
			rms=rmse(result,outDepth);
			mad_ =mad(result,outDepth);
            % imwrite(pree,sprintf('%s/%s-%.3f.png',result_path,filepaths(i).name(1:end-4),rms));
            % save(sprintf('%s/SRX%d-epoch-%d-rms-%.3f-mad-%.3f.mat',result_path,scale,epoch,rms,mad_),'softmax_pre','result');
            save(sprintf('%s/SRX%d-epoch-%d-rms-%.3f-mad-%.3f.mat',result_path,scale,epoch,rms,mad_),'result','softmax_pre','noisy_depth','outDepth');
            % save(sprintf('%s/%s.mat',result_path,filepaths(i).name(1:end-4)),'pre1','outDepth');
        end
    end
end
