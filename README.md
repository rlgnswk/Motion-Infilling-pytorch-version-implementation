# Implementation: Convolutional Autoencoders for Human Motion Infilling(Pytorch)
Unofficial personal code
Note: I added batch normalization for stable training.


-----------------

## Result:

![Result1](./fig/imple_result.gif)

----------
# Usage:

##Data
1. Get data from [Here](https://theorangeduck.com/page/deep-learning-framework-character-motion-synthesis-and-editing)
2. Prepocessing by using code from official [Github](https://github.com/assafshocher/ZSSR](https://github.com/eth-ait/motion-infilling/tree/be814cfe971ec58d0e66c7644db3cdc89f71d092)



## Run on your data:
You can find additional dataset 
from [Here](https://drive.google.com/file/d/16L961dGynkraoawKE2XyiCh4pdRS-e4Y/view) 
provided by [MZSR](https://github.com/JWSoh/MZSR) (CVPR 2020)

First, put your data files in ```<ZSRGAN_path>/datasets/```

The results will save in ```<ZSRGAN_path>/experiments/```

```
python train.py --name <save_result_path> --dataset <name_of_your_dataset> --GT_path <HR_folder_in_your_dataset> --LR_path <LR_folder_in_your_dataset>
```


# Reference
Reference [Paper](https://arxiv.org/abs/2010.11531)(3DV 2020)
Official Code(Tensorflow) [Github](https://github.com/assafshocher/ZSSR](https://github.com/eth-ait/motion-infilling/tree/be814cfe971ec58d0e66c7644db3cdc89f71d092) 

