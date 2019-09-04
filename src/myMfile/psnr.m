function out = psnr(im,gt)
if isa(gt,'uint8')
    gt=im2double(gt);
    im=im2double(im);
end
out = -10*log10( mean( (im(:)-gt(:)).^2 ) );