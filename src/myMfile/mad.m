function y=mad(gt,img)
if isa(gt,'uint8')
    gt=double(gt);
    img=double(img);
else
    gt=double(uint8(gt*255));
    img=double(uint8(img*255));
end
n=size(gt,1)*size(gt,2);
res=abs(gt-img);
y=sum(res(:))/n;