function net= jointNet(netclass,netSR)

    net= dagnn.DagNN.loadobj(netclass);
    net.renameLayer('loss','edgeloss');
    net.renameVar('loss','edgeloss');
    net.renameVar('input_label','input_elabel');
    net.addLayer('softmax',dagnn.SoftMax(),'predict_conv','softmax');
    net.addLayer('slice',dagnn.Slice('thres',0.3),'softmax','edge');
%     net2=dagnn.DagNN();


net.addLayer('depthedgehfre',dagnn.Concat(),{'input_d','edge','input_g'},'depthedgehfre');
netSR.layers=netSR.layers(1:end-1);
% net.addFromSimpleNN(netSR, 'canonicalNames', true) ;
net=addFromSimpleNN(net,netSR) ;
net.setLayerInputs('layer1',{'depthedgehfre'});
net.addLayer('SRloss',...
    dagnn.lpLoss('lp',0.2),...
    {'x13','input_dlabel','instance_weights'},'SRloss',{});

	net2=net;
	net=net2.saveobj();
	
	save('net-init.mat', 'net');
	net=net2;
% for i=1:6
% net.renameVar(sprintf('x%d',i),sprintf('layerConv%d',i));
% end
% net2.setLayerOutputs('layer1',{'predict'});

end