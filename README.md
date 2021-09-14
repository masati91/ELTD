# EdgeLess Layer for Recyclable Trash Detection

![ELTD](./docs/model_Architecture.jpg)


## ntroduction

We propose a novel anchor-free model with an edgeless kernel for recognizing and classifying complex unstructured recyclables, and a recyclables dataset required for model training. 

First, we create a recyclable dataset according to the resource separation and emission standards set by the Korean Ministry of Environment. Also, We define a class for waste that the general public should recycle and strengthen the annotation. 

Second, the proposed edgeless module consists of two types: Background Noise Reduce Module for correcting the feature map of the backbone and an Instance separation module for correcting the feature map of the head. Background Noise Reduce Module for the feature map of the backbone corrects the edges of the entire feature map of the input image. 

The Instance separation module for the feature map of head corrects the edges of each instance contained in the image. The result is improved detection accuracy for overlapping or small wastes. Our model with an Edgeless module shows optimal performance in recyclable recognizing. 

And we shall show that the AP score is improved by 3.9\% and the F1 score by more than 2\% compared to the latest models of the one-stage detector and compare the performance with other latest models.


The repo is based on **[mmdetection](https://github.com/open-mmlab/mmdetection)**.



## Benchmark


|Model          |    Backbone     |    MS  |  Rotate | Lr schd  | Inf time (fps) | box AP (ori./now) | Download|
|:-------------:| :-------------: | :-----:| :-----: | :-----:  | :------------: | :----: | :---------------------------------------------------------------------------------------: |
|RetinaNet      |    R-50-FPN     |   -     |   -    |   1x     |      16.0      |  68.05/68.40 |        [model](https://drive.google.com/file/d/1ZUc8VUDOkTnVA1FFNuINm2U39h0anLPm/view?usp=sharing)        |
|S<sup>2</sup>A-Net         |    R-50-FPN     |   -     |   -    |   1x     |      16.0      |  74.12/73.99|    [model](https://drive.google.com/file/d/19gwDSzCx0uToqI9LyeAg_yXNLgK3sbl_/view?usp=sharing)    |
|S<sup>2</sup>A-Net         |    R-50-FPN     |   ✓     |  ✓     |   1x     |      16.0      |  79.42 |    [model](https://drive.google.com/file/d/1W-JPfoBPHdOxY6KqsD0ZhhLjqNBS7UUN/view?usp=sharing)    |
|S<sup>2</sup>A-Net         |    R-101-FPN    |   ✓     |  ✓     |   1x     |      12.7      |  79.15 |    [model](https://drive.google.com/file/d/1Jkbx-WvKhokEOlWR7WLKxTpH4hDTp-Tb/view?usp=sharing)            |

*Note that the mAP reported here is a little different from the original paper. All results are reported on DOTA-v1.0 *test set*. 
All checkpoints here are trained with the [Original version](https://github.com/csuhan/s2anet/tree/original_version), and **not compatible** with the updated version.


## Installation

Please refer to [install.md](install.md) for installation and dataset preparation.


## Getting Started

Please see [getting_started.md](get_started.md) for the basic usage of MMDetection.



## Citation

```
@article{kang2021,  
  author={BoSeon Kang, ChangSeong Jeong},  
  journal={},   
  title={Edgeless Layer for Recyclable Trash Detection},   
  year={2021}, 
  pages={},  
  doi={}}
```
