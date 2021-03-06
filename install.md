## Prerequisites

- Linux or macOS (Windows is in experimental support)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

Compatible MMDetection and MMCV versions are shown as below. Please install the correct version of MMCV to avoid installation issues.

| MMDetection version |    MMCV version     |
|:-------------------:|:-------------------:|
| 2.10.0               | mmcv-full>=1.2.4, <1.4.0 |

**Note:** You need to run `pip uninstall mmcv` first if you have mmcv installed.
If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

## Installation

### Prepare environment

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n ELTD python=3.7
    conda activate ELTD
    ```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

    ```shell
    conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
    ```
    

### Install MMDetection

It is recommended to install MMDetection with [MIM](https://github.com/open-mmlab/mim),
which automatically handle the dependencies of OpenMMLab projects, including mmcv and other python packages.

```shell
pip install openmim
mim install mmdet
```

Or you can still install MMDetection manually:

1. Install mmcv-full.

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
    ```


2. Install MMDetection.

    You can simply install mmdetection with the following command:


    ```shell
    git clone -b v2.12.0 https://github.com/open-mmlab/mmdetection.git
    cd mmdetection
    pip install -r requirements/build.txt
    pip install -v -e .  # or "python setup.py develop"
    ```
    

**Note:**

a. When specifying `-e` or `develop`, MMDetection is installed on dev mode
, any local modifications made to the code will take effect without reinstallation.

b. If you would like to use `opencv-python-headless` instead of `opencv-python`,
you can install it before installing MMCV.

c. Some dependencies are optional. Simply running `pip install -v -e .` will
 only install the minimum runtime requirements. To use optional dependencies like `albumentations` and `imagecorruptions` either install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -v -e .[optional]`). Valid keys for the extras field are: `all`, `tests`, `build`, and `optional`.

d. If you would like to use `albumentations`, we suggest using
`pip install albumentations>=0.3.2 --no-binary imgaug,albumentations`. If you simply use
`pip install albumentations>=0.3.2`, it will install `opencv-python-headless` simultaneously (even though you have already installed `opencv-python`). We should not allow `opencv-python` and `opencv-python-headless` installed at the same time, because it might cause unexpected issues. Please refer to [official documentation](https://albumentations.ai/docs/getting_started/installation/#note-on-opencv-dependencies) for more details.

### Install without GPU support

MMDetection can be built for CPU only environment (where CUDA isn't available).

In CPU mode you can run the demo/webcam_demo.py for example.
However some functionality is gone in this mode:

- Deformable Convolution
- Modulated Deformable Convolution
- ROI pooling
- Deformable ROI pooling
- CARAFE: Content-Aware ReAssembly of FEatures
- SyncBatchNorm
- CrissCrossAttention: Criss-Cross Attention
- MaskedConv2d
- Temporal Interlace Shift
- nms_cuda
- sigmoid_focal_loss_cuda
- bbox_overlaps

If you try to run inference with a model containing above ops, an error will be raised.
The following table lists affected algorithms.

|                        Operator                         |                            Model                             |
| :-----------------------------------------------------: | :----------------------------------------------------------: |
| Deformable Convolution/Modulated Deformable Convolution | DCN???Guided Anchoring???RepPoints???CentripetalNet???VFNet???CascadeRPN???NAS-FCOS???DetectoRS |
|                      MaskedConv2d                       |                       Guided Anchoring                       |
|                         CARAFE                          |                            CARAFE                            |
|                      SyncBatchNorm                      |                           ResNeSt                            |

**Notice:** MMDetection does not support training with CPU for now.

### Another option: Docker Image

We provide a [Dockerfile](https://github.com/open-mmlab/mmdetection/blob/master/docker/Dockerfile) to build an image. Ensure that you are using [docker version](https://docs.docker.com/engine/install/) >=19.03.

```shell
# build an image with PyTorch 1.6, CUDA 10.1
docker build -t mmdetection docker/
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection/data mmdetection
```

### A from-scratch setup script

Assuming that you already have CUDA 10.1 installed, here is a full script for setting up MMDetection with conda.

```shell
conda create -n openmmlab python=3.7 -y
conda activate openmmlab

conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y

# install the latest mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

# install mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
```

### Developing with multiple MMDetection versions

The train and test scripts already modify the `PYTHONPATH` to ensure the script use the MMDetection in the current directory.

To use the default MMDetection installed in the environment rather than that you are working with, you can remove the following line in those scripts

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

## Verification

To verify whether MMDetection is installed correctly, we can run the following sample code to initialize a detector and inference a demo image.

```python
from mmdet.apis import init_detector, inference_detector

config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
inference_detector(model, 'demo/demo.jpg')
```

The above code is supposed to run successfully upon you finish the installation.
