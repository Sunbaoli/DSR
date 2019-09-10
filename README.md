# Depth Super-Resolution based on Deep Edge-Aware Learning

This repo implements the training and testing of depth upsampling networks for "Depth Super-Resolution based on Deep Edge- Awar Learning" by Xinchen Ye, Baoli Sun, and et al. at DLUT.

## The proposed edge-guided depth super-resolution framework
![](https://github.com/Sunbaoli/DSR/blob/master/code/fig2.png)

## Results
![](https://github.com/Sunbaoli/DSR/blob/master/code/fig1.png)


This repo can be used for training and testing of depth upsampling under noiseless and noisy cases for Middleburry  datasets. Some trained models are given to facilitate simple testings before getting to know the code in detail. Besides,  the results of our inferred edge maps, recovered depth maps under both noiseless and noisy cases are all given to make it  easy to compare with and reference our work.

## Dependences

matlab r2017a

matconvnet-1.0-beta25

## Train
` run start_train.m `

## Test
` run test_classSR.m `

Testing on Middlebury noisy depth maps, you can modify the ` 'test_classSR.m' ` in lines 68:
` im_depth=imnoise(imresize(im_label,1/scale,method),'gaussian',0,(5/255)^2);noisy_depth=im_depth; `


## Citation 
If you find this code useful, please cite:

` Xinchen Ye* et al., Depth Super-Resolution based on Deep Edge-Aware Learning, Submitted to Pattern Recognition, Major revision. `


