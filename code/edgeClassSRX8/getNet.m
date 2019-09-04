function net=getNet()
if exist('net-init.mat')
load('net-init.mat', 'net');
    % for i=1:length(net.layers)
        % if strcmp(net.layers(i).type,'dagnn.ReLU')
            % net.layers(i).block.leak=0.01;
        % end
    % end
    net= dagnn.DagNN.loadobj(net);

else

load('netclass.mat','net');
netclass=net;
load('netSR.mat','net');
netSR=net;
sz=size(netSR.layers{1}.weights{1});
if sz(1)==1
    netSR.layers{1}.weights{1}(:,:,2,:)=0;
% else 
    % netSR.layers{1}.weights{1}(:,:,,:)=0;
end
net=jointNet(netclass,netSR);
end