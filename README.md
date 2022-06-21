# Implementation: Convolutional Autoencoders for Human Motion Infilling(Pytorch)
### Unofficial personal code

### Note: I added batch normalization for stable training.
### Reference
Reference [Paper]([[https://github.com/assafshocher/ZSSR](https://github.com/eth-ait/motion-infilling/tree/be814cfe971ec58d0e66c7644db3cdc89f71d092](https://arxiv.org/abs/2010.11531)))(3DV 2020)
Official Code(Tensorflow) [Github]([https://github.com/assafshocher/ZSSR](https://github.com/eth-ait/motion-infilling/tree/be814cfe971ec58d0e66c7644db3cdc89f71d092)) 
-----------------

## Result:

![Result1](./fig/imple_result.gif)

----------
# Usage:

## Run on sample data:
First, the sample data(Degraded Set5) already are placed in ```<ZSRGAN_path>/datasets/MySet5```

The results will save in ```<ZSRGAN_path>/experiments/```

```
python train.py --name <save_result_path>
```
## Run on your data:
You can find additional dataset 
from [Here](https://drive.google.com/file/d/16L961dGynkraoawKE2XyiCh4pdRS-e4Y/view) 
provided by [MZSR](https://github.com/JWSoh/MZSR) (CVPR 2020)

First, put your data files in ```<ZSRGAN_path>/datasets/```

The results will save in ```<ZSRGAN_path>/experiments/```

```
python train.py --name <save_result_path> --dataset <name_of_your_dataset> --GT_path <HR_folder_in_your_dataset> --LR_path <LR_folder_in_your_dataset>
```
# References
Official tensorfolw version [Github]([https://github.com/assafshocher/ZSSR](https://github.com/eth-ait/motion-infilling/tree/be814cfe971ec58d0e66c7644db3cdc89f71d092)) (3DV 2020)
