%%%%loss by dxy
classdef dLoss < dagnn.Loss
    properties
        lp = 1
    end
    methods
        function outputs = forward(obj, inputs, params)
            if obj.lp==2
                d=bsxfun(@minus, inputs{1}, inputs{2});
                outputs{1} = sum(d(:).^2);
            end
            if obj.lp==1
                delta = inputs{1} - inputs{2} ;
                absDelta = abs(delta) ;
                outputs{1} = inputs{3}(:)' * absDelta(:) ;
            end
            n = obj.numAveraged ;
            m = n + gather(sum(inputs{3}(:))) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            if obj.lp==2
                d=bsxfun(@minus, inputs{1}, inputs{2});
                derInputs={d .* derOutputs{1}, [], []};
                derParams=1;
            end
            if obj.lp==1
                delta = inputs{1} - inputs{2} ;
                %                 absDelta = abs(delta) ;
                derInputs = {inputs{3} .* sign(delta) .* derOutputs{1}, [], []} ;
                derParams = {} ;
            end
        end
        
        function obj = dLoss(varargin)
            obj.load(varargin) ;
            obj.loss = 'dLoss';
        end
    end
end