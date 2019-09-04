classdef LossSmoothL1 < dagnn.Loss
    %LossSmoothL1  Smooth L1 loss
    %  `LossSmoothL1.forward({x, x0, w})` computes the smooth L1 distance
    %  between `x` and `x0`, weighting the elements by `w`.
    %
    %  Here the smooth L1 loss between two vectors is defined as:
    %
    %     Loss = sum_i f(x_i - x0_i) w_i.
    %
    %  where f is the function (following the Faster R-CNN definition):
    %
    %              { 0.5 * sigma^2 * delta^2,         if |delta| < 1 / sigma^2,
    %   f(delta) = {
    %              { |delta| - 0.5 / sigma^2,         otherwise.
    %
    %  In practice, `x` and `x0` can pack multiple instances as 1 x 1 x C
    %  x N arrays (or simply C x N arrays).
    
    properties
        sigma = 1.
        lp=0  %by dxy
        w=[1 1]  %by dxy
		
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            sigma2 = obj.sigma^2 ;
            delta = inputs{1} - inputs{2} ;
            absDelta = abs(delta) ;
            
            linearRegion = (absDelta > 1. / sigma2) ;
            absDelta(linearRegion) = absDelta(linearRegion) - 0.5/sigma2 ;
            absDelta(~linearRegion) = 0.5 * sigma2 * absDelta(~linearRegion).^2 ;
			lossLp=0;
            if obj.lp
                lossLp=vl_nnpdist( inputs{1}, inputs{2},obj.lp,'noRoot',true,'aggregate',true);
            end
            % Mutliply by instance weights and sum.
            outputs{1} = obj.w(1)*inputs{3}(:)' * absDelta(:)+obj.w(2)*lossLp ;
            
            % outputs{1} = inputs{3}(:)' * absDelta(:) ;
            % Accumulate loss statistics.
            n = obj.numAveraged ;
            m = n + gather(sum(inputs{3}(:))) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            % Function derivative:
            %
            %          { sigma^2 * x,             if |x| < 1 / sigma^2,
            %  f'(x) = {
            %          { sign(x),                 otherwise.
            
            sigma2 = obj.sigma^2 ;
            delta = inputs{1} - inputs{2} ;
            absDelta = abs(delta) ;
            
            linearRegion = (absDelta > 1. / sigma2) ;
            delta(linearRegion) = sign(delta(linearRegion));
            delta(~linearRegion) = sigma2 * delta(~linearRegion) ;
			lossLp=0;
            if obj.lp
                derlossLp=vl_nnpdist( inputs{1}, inputs{2},obj.lp,derOutputs{1},'noRoot',true,'aggregate',true);
            end
            derInputs = {obj.w(1)*inputs{3} .* delta .* derOutputs{1}+obj.w(2)*derlossLp, [], []} ;
            % derInputs = {inputs{3} .* sign(delta) .* derOutputs{1}, [], []} ;
            derParams = {} ;
        end
        
        function obj = LossSmoothL1(varargin)
            obj.load(varargin) ;
            obj.loss = 'smoothl1';
			
        end
    end
end
