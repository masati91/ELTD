  
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from random import randint
import random
import cv2
import mmcv
import numpy as np
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmcv.utils import print_log
from mmcv.image import imread, imwrite

from mmdet.utils import get_root_logger


class PostProcessor():
    def __init__(self,
                classes,
                score_thr=0.3,
                #thickness=1,
                thickness=1,
                #font_scale=0.5,
                font_scale=0.5,
                win_name='',
                wait_time=0):
        self.num_classes = len(classes)
        self.classes = classes
        self.score_thr = score_thr
        self.thickness = thickness
        self.font_scale = font_scale
        self.win_name = win_name
        self.wait_time = wait_time
        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255),
                        (255, 0, 255), (255, 255, 255), (0, 0, 0), (255, 0, 128), (0, 191, 255),
                        (10, 255, 128), (191, 255, 0), (255, 191, 0), (255, 128, 10), (50, 152, 89)]
        self.makeColors()

    def makeColors(self):
        if len(self.colors) >= self.num_classes:
            return
        else:
            while len(self.colors) < self.num_classes:
                self.colors.append((randint(20, 230), random.randint(20, 230), random.randint(20, 230)))
            return

    def saveResult(self,
                    img,
                    result,
                    show=False,
                    out_file=None):
        img, bboxes, labels = self.extractInfo(img, result, show=False, out_file=out_file)

        # draw bounding boxes
        return self.imshow_det_bboxes(img, bboxes, labels, show=show, out_file=out_file)


    def bb_intersection_over_union(self, bboxes, labels, box_scores):
        # determine the (x, y)-coordinates of the intersection rectangle
        best_indexes = []

        for i in range(0, len(bboxes) - 1):

            best_iou = -1
            best_list = []

            for j in range(i + 1 , len(bboxes)):
                xA = max(bboxes[i][0], bboxes[j][0])
                yA = max(bboxes[i][1], bboxes[j][1])
                xB = min(bboxes[i][2], bboxes[j][2])
                yB = min(bboxes[i][3], bboxes[j][3])
                # compute the area of intersection rectangle
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                # compute the area of both the prediction and ground-truth
                # rectangles
                boxAArea = (bboxes[i][2] - bboxes[i][0] + 1) * (bboxes[i][3] - bboxes[i][1] + 1)
                boxBArea = (bboxes[j][2] - bboxes[j][0] + 1) * (bboxes[j][3] - bboxes[j][1] + 1)
                # compute the intersection over union by taking the intersection
                # area and dividing it by the sum of prediction + ground-truth
                # areas - the interesection area
                iou = interArea / float(boxAArea + boxBArea - interArea)
                if iou > best_iou:
                    best_iou = iou
                    best_list = [i , j, best_iou]

                    best_indexes.append(best_list)

        index = []
        for best_index in best_indexes:
            if best_index[2] > 0.98: # best_iou
                if box_scores[best_index[0]] > box_scores[best_index[1]]:
                    index.append(best_index[1])

                else :
                    index.append(best_index[0])

        index = set(index)
        index = sorted(list(index), reverse=True)

        for i in index :
            if box_scores[i] < 0.5:
                bboxes = np.delete(bboxes, i, axis = 0)
                labels = np.delete(labels, i, axis = 0)
                box_scores = np.delete(box_scores, i, axis = 0)


        # return the intersection over union value
        return bboxes, labels, box_scores

    def cropBoxes(self, img, result, out_file=None):
        img, bboxes, labels = self.extractInfo(img, result, show=False, out_file=out_file)
        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert bboxes.shape[0] == labels.shape[0]
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
        img = imread(img)
        box_scores = []

        if self.score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > self.score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            box_scores = scores[inds]

        img = np.ascontiguousarray(img)
        croppedImgs = []
        out_label = []

        if len(labels) > 1:
            bboxes, labels, box_scores = self.bb_intersection_over_union(bboxes, labels, box_scores)

        # path to save cropped image if save
        # splitPath = out_file.split('/')
        # fileName = splitPath.pop(-1).split('.')[0]

        for idx, (bbox, label) in enumerate(zip(bboxes, labels)):

            # !!!!!!!!!!! ~ Except class cap(8) or label(9) ~ !!!!!!!!!!!!
            if label != 8 and label != 9:
                
                bbox_int = bbox.astype(np.int32)
                heightRange = (bbox_int[1], bbox_int[3])
                widthRange = (bbox_int[0], bbox_int[2])

                dst = img.copy()

                center_x = int(int(bbox_int[0]) - int(bbox_int[0])*0.15)
                center_y = int(int(bbox_int[1]) - int(bbox_int[0])*0.15)
                width = int(int(bbox_int[2]) + int(bbox_int[2])*0.15)
                height = int(int(bbox_int[3]) + int(bbox_int[3])*0.15)

                dst = dst[center_y:height, center_x:width]

                # dst = dst[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]]

                croppedImgs.append(dst)

            out_label.append(label)

            # save cropped image            
            # out_file = splitPath.copy()
            # out_file.append(fileName+'_'+str(idx)+'.jpg')
            # out_file = '/'.join(out_file)
            # if out_file is not None:
            #     imwrite(dst, out_file)

        return croppedImgs, out_label

    def extractInfo(self,
                    img,
                    result,
                    show=False,
                    out_file=None):
        """Draw `result` over `img`.
        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > self.score_thr)[0]
            np.random.seed(42)
            color_masks = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            for i in inds:
                i = int(i)
                color_mask = color_masks[labels[i]]
                mask = segms[i].astype(bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        # if not (show or out_file):
        #     return img

        return img, bboxes, labels


    def imshow_det_bboxes(self,
                        img,
                        bboxes,
                        labels,
                        show=True,
                        out_file=None):
        """Draw bboxes and class labels (with scores) on an image.
        Args:
            img (str or ndarray): The image to be displayed.
            bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
                (n, 5).
            labels (ndarray): Labels of bboxes.
            class_names (list[str]): Names of each classes.
            score_thr (float): Minimum score of bboxes to be shown.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            show (bool): Whether to show the image.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
            out_file (str or None): The filename to write the image.
        Returns:
            ndarray: The image with bboxes drawn on it.
        """
        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert bboxes.shape[0] == labels.shape[0]
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
        img = imread(img)

        if self.score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > self.score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]

        img_file_name = out_file.split("/")[-1]

        img = np.ascontiguousarray(img)
        for bbox, label in zip(bboxes, labels):
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            cv2.rectangle(
                img, left_top, right_bottom, self.colors[label], thickness=self.thickness, lineType=cv2.LINE_AA)
            label_text = self.classes[
                label] if self.classes is not None else f'cls {label}'

            if len(bbox) > 4:
                label_text += f' |{bbox[-1]:.02f}'

            cv2.putText(img, label_text, (bbox_int[0], bbox_int[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.colors[label], lineType=cv2.LINE_AA)

            ### Bounding Box Save

            # f = open(os.path.join("results/detectors_padding_2/gt/" , "gt_" + img_file_name.split(".")[0] + ".txt"), 'a')
            # f.write(str(int(bbox[0])) + "," + str(int(bbox[1])) + "," + 
            #         str(int(bbox[2])) + "," + str(int(bbox[1])) + "," + 
            #         str(int(bbox[2])) + "," + str(int(bbox[3])) + "," + 
            #         str(int(bbox[0])) + "," + str(int(bbox[3])) + "," + str(self.classes[label]) + "\n")

        if show:
            imshow(img, self.win_name, self.wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img
