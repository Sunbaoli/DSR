classdef lpLoss < dagnn.Loss
    properties
        lp = 0.2
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            t = ((inputs{1}-inputs{2}).^2)/2;
            % lossLp=vl_nnpdist( inputs{1}, inputs{2},obj.lp,'noRoot',true,'aggregate',true);
            lossLp=vl_nnpdist( inputs{1}, inputs{2},obj.lp,'noRoot',true,'aggregate',false);
            Y = t +lossLp;
            outputs{1} = inputs{3}(:)' * Y(:) ;
            % Accumulate loss statistics.
            n = obj.numAveraged ;
            m = n + gather(sum(inputs{3}(:))) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            
            % derInputs = {bsxfun(@minus,inputs{1},inputs{2}).*derOutputs{1}+	vl_nnpdist( inputs{1},inputs{2},obj.lp,derOutputs{1},'noRoot',true,'aggregate',true),[],[]};
            derInputs = {bsxfun(@minus,inputs{1},inputs{2}).*derOutputs{1}+	vl_nnpdist( inputs{1},inputs{2},obj.lp,derOutputs{1},'noRoot',true,'aggregate',false),[],[]};
                        derParams = {} ;
        end
        
        function obj = lpLoss(varargin)
            obj.load(varargin) ;
            obj.loss = 'lpLoss';
        end
    end
end

