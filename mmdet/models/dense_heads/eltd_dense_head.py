from abc import ABCMeta, abstractmethod

import torch.nn as nn

from torchvision.utils import make_grid
import torch
import copy
from mmdet.core import multi_apply
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


class ELTDBaseDenseHead(nn.Module, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self):

        super(ELTDBaseDenseHead, self).__init__()

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    @abstractmethod
    def get_bboxes(self, **kwargs):
        """Transform network output for a batch into bbox predictions."""
        pass

    def tensortoimg(self, x, img_metas):
        print(img_metas)

        for k in range(0, len(img_metas)):
            for i in range(0, 1):
                feat_show = x[i][k]
                for j in range(0, 10): #len(feat_show)):
                    feat_save = feat_show[j]
                    feat_save = feat_save.cpu().detach().numpy()
                    plt.imsave("/home/ELTD/tools/featuremap/FPN_" +str(i) + "_" + str(j) + "_" + str(k) + ".jpg", feat_save)

    def get_epanechnikov_kernel_1d(self, size, sigma):

        arr = np.arange(math.trunc(size/2)*(-1), math.ceil(size/2)+1 ,1) # All use 
        kernel = 1 - ((arr*arr)/(sigma*sigma))

        return kernel[1:]

    def get_gaussian_filter_1d(self, size, sigma):

        arr = np.arange(math.trunc(size/2)*(-1), math.ceil(size/2)+1 ,1) 

        kernel = np.exp((-arr*arr)/(2*sigma*sigma)) 
        
        return kernel[:-1]

    def get_cosine_kernel_1d(self, size, sigma):

        arr = np.arange(math.trunc(size/2)*(-1), math.ceil(size/2)+1 ,1) # All use 
        s = np.abs((sigma*sigma) / ( 1/3 - 2/np.pi*np.pi))
        kernel_raw = (1/2*s) * ( 1 + np.cos(arr * np.pi / s))
        kernel = kernel_raw/kernel_raw.sum() 

        return kernel[:-1]

    def get_masati_kernel_1d(self, size):
        if size < 3:
            kernel = np.ones(size)

        else :
            arr = np.ones(size)
            distance_value = 2 / (size - 1)
            for idx in range(0, len(arr)):
                arr[idx] = arr[idx] - (distance_value * idx)
            kernel = np.exp(-abs(arr/0.85)**8)

        return kernel


    def featuremap_distribution(self, x, img_metas, gt_bboxes, kernel_method, kernel_sigma):
        num_imgs = len(img_metas)

        distributionmaps = [ [ 0 for col in range(num_imgs)] for row in range(5)]

        each_level_gt_bboxes = copy.deepcopy(gt_bboxes)
        # kernel_method = 'cv_gaussian'
        
        for j in range(0, 5):
            for i, img_meta in enumerate(img_metas):
                # Featuremap size tensor generate
                featmap_size = x[j][i].size()[-2:]
                featmap_device = x[j][i].device
                distributionmaps[j][i] = torch.ones((featmap_size[0], featmap_size[1]), device=featmap_device)
                # GT box size convert
                img_size = img_meta['ori_shape']

                x_sacle_value = round( img_size[0] / featmap_size[0] )
                y_sacle_value = round( img_size[1] / featmap_size[1] )

                for index in range(0,4):
                    if index % 2 == 0:
                        each_level_gt_bboxes[i][:, index] = gt_bboxes[i][:, index] / x_sacle_value
                    else :
                        each_level_gt_bboxes[i][:, index] = gt_bboxes[i][:, index] / y_sacle_value
            
                # Apply each_level_gt_bboxes to distributionmap 
                for each_level_gt_bbox in each_level_gt_bboxes[i]:

                    kernel = distributionmaps[j][i][int(each_level_gt_bbox[1]):int(each_level_gt_bbox[3]), 
                                                    int(each_level_gt_bbox[0]):int(each_level_gt_bbox[2])]

                    kernel_size = kernel.size()

                    bbox_width = kernel_size[0]
                    bbox_hight = kernel_size[1]

                    if kernel_method == 'gaussian_kernel':
                        kernel_x = cv2.getGaussianKernel(bbox_width if bbox_width >= 1 else 1, kernel_sigma)
                        kernel_y = cv2.getGaussianKernel(bbox_hight if bbox_hight >= 1 else 1, kernel_sigma)

                        kernel = np.outer(kernel_x, kernel_y)

                        value = 1 / kernel.max()
                        kernel = np.clip(kernel*value, 0.3, 1)
                        
                    elif kernel_method == 'epanechnikov_kernel':
                        kernel_x = self.get_epanechnikov_kernel_1d(bbox_width if bbox_width >= 1 else 1, bbox_width*0.75)
                        kernel_y = self.get_epanechnikov_kernel_1d(bbox_hight if bbox_hight >= 1 else 1, bbox_hight*0.75)

                        kernel = np.outer(kernel_x, kernel_y)

                    elif kernel_method == 'gaussian2_kernel':
                        kernel_x = self.get_gaussian_filter_1d(bbox_width if bbox_width >= 1 else 1, kernel_sigma)
                        kernel_y = self.get_gaussian_filter_1d(bbox_hight if bbox_hight >= 1 else 1, kernel_sigma)

                        kernel = np.outer(kernel_x, kernel_y)

                    elif kernel_method == 'cosine_kernel':
                        kernel_x = self.get_cosine_kernel_1d(bbox_width if bbox_width >= 1 else 1, kernel_sigma)
                        kernel_y = self.get_cosine_kernel_1d(bbox_hight if bbox_hight >= 1 else 1, kernel_sigma)

                        kernel = np.outer(kernel_x, kernel_y)

                        value = 1 / kernel.max()
                        kernel = np.clip(kernel*value, 0.001, 1)

                    elif kernel_method == 'masati_kernel':
                        kernel_x = self.get_masati_kernel_1d(bbox_width if bbox_width >= 1 else kernel_sigma)
                        kernel_y = self.get_masati_kernel_1d(bbox_hight if bbox_hight >= 1 else kernel_sigma)

                        kernel = np.outer(kernel_x, kernel_y)
                        kernel = np.clip(kernel, 0.1, 1)


                    distributionmaps[j][i][int(each_level_gt_bbox[1]):int(each_level_gt_bbox[3]), 
                                          int(each_level_gt_bbox[0]):int(each_level_gt_bbox[2])] = torch.from_numpy(kernel)


        # Apply distributionmap to Featuremap (x) 
        list_x = list(x)
        for idx, (each_x, distributionmap) in enumerate(zip(list_x, distributionmaps)):
            for i, each_distributionmap in enumerate(distributionmap):
                if i == 0:
                    out_x = each_distributionmap.unsqueeze(0).unsqueeze(0)
                else:
                    out_x = torch.cat([out_x, each_distributionmap.unsqueeze(0).unsqueeze(0)], dim=0)
            list_x[idx] = list_x[idx].mul(out_x)
        x = tuple(list_x)
        return x

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        # tensor save
        self.tensortoimg(x, img_metas)

        x = self.featuremap_distribution(x, img_metas, gt_bboxes, self.kernel_method, self.kernel_sigma)

        self.tensortoimg(x, img_metas)
        outs = self(x)

        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list
