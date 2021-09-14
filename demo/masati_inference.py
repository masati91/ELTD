from argparse import ArgumentParser
import os
from os import path, listdir

import re

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

from hamsters_utils import PostProcessor

from tqdm import tqdm

classes = ['paper', 'paperpack', 'papercup', 'can', 'bottle', 'pet', 'plastic', 'vinyl', 'cap', 'label']

def makeImgList(dir):
    imgFileNames = listdir(dir)
    imgFileNames = [x for x in imgFileNames if x.endswith(('jpg', 'JPG', 'jpeg', 'JPEG'))]
    imgFileNames = [x for x in imgFileNames]

    return imgFileNames

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


def main():
    parser = ArgumentParser()
    parser.add_argument('--imgpath', default="data/coco/val2017", help='Image file')
    parser.add_argument('--savepath', default="output/", help='Image file')
    parser.add_argument('--config', default="configs/eltd/eltd_r50_fpn_1x_coco.py", help='Config file')
    parser.add_argument('--checkpoint', default="eltd_r50_fpn_1x_epoch_12.pth", help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument('--imgsave', default=True, help='image save')
    parser.add_argument('--csvsave', default=True, help='csv file save')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image

    for file in os.scandir(args.savepath):
        os.remove(file)

    image_list = sorted_aphanumeric(makeImgList(args.imgpath))

    f = open("demo/answer.csv", 'w')
    f.write("file_name,paper,paperpack,papercup,can,bottle,pet,plastic,vinyl,cap,label \n")

    for img in tqdm(image_list):

        image_file = args.imgpath + "/" + img

        result = inference_detector(model, image_file)
        # show the results
        # show_result_pyplot(model, args.img, result, score_thr=args.score_thr)

        # PostProcessor init
        detectorsPostProcessor = PostProcessor(classes, score_thr=args.score_thr)

        if args.csvsave :
            f.write(img)

            output_class = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            _, labels = detectorsPostProcessor.cropBoxes(image_file, result, out_file=(args.savepath))

            for label in labels:
                output_class[label] = 1

            for i in output_class:
                f.write("," + str(i))
            f.write("," + "\n")

        if args.imgsave :
            detectorsPostProcessor.saveResult(image_file, result, show=False, out_file=(args.savepath + img))


if __name__ == '__main__':
    main()
