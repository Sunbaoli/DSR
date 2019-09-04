function y=rmse(gt,im,origin)
% if origin==true,compare in origin value
% else consider as an image, in unit8 0-255
if nargin<3
origin=false;
end
if origin
    gt=double(gt);
    im=double(im);
else
    if ~isa(gt,'uint8')
        gt=double(uint8(gt*255));
        im=double(uint8(im*255));
    else
        gt=double(gt);
        im=double(im);
    end
end

n=size(gt,1)*size(gt,2);
square=((gt-im).^2)/n;
y=sqrt(sum(square(:)));
end