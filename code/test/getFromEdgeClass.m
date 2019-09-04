function result=getFromEdgeClass(score_map,classNum)
% get the class result

index=bsxfun(@eq,score_map,max(score_map,[],3));
C=1:classNum;
C=reshape(C,[1 1 classNum]);
index =bsxfun(@times,index,C);
result = sum(index,3);
end
