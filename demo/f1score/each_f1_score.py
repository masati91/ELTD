import csv
import time
import os
import numpy as np
import argparse
import re
from sklearn.metrics import f1_score


def get_arguments():
    parser = argparse.ArgumentParser(description='IITP_score')
    parser.add_argument('--gt_file', type=str, default='sample_GT.csv')
    parser.add_argument('--pred_file', type=str, default='answer.csv')

    return parser.parse_args()


def read_file(gt_file, pred_file):
    gt = []
    predict = []
    with open(gt_file, newline='', encoding='utf-8') as gtf, open(pred_file, newline='', encoding='utf-8') as predf:
        gt_reader = csv.reader(gtf)
        gt_header = next(gt_reader)
        pred_reader = csv.reader(predf)
        pred_header = next(pred_reader)
        for gt_row in gt_reader:
            gt.append(gt_row)

        for pred_row in pred_reader:
            predict.append(pred_row)

    return gt, predict


def scoring(gt, pred, j):
    n = len(gt)-1
    score = 0
    
    temp_gt = []
    temp_pred = []

    for i in range(n):

        temp_gt.append(int(gt[i][j]))
        temp_pred.append(int(pred[i][j]))

        score += f1_score(temp_gt, temp_pred, average="macro", zero_division=0)


    return (score / n)*100.0


def main():
    args = get_arguments()
    gt, pred = read_file(args.gt_file, args.pred_file)
    label_list = ["background", "paper    ", "paperpack", "papercup ", "can      ", "bottle   ", "pet      ", "plastic  ", "vinyl    ","cap      ","label    "]
    for i in range(1, 11):
        results = scoring(gt, pred, i)
        print( label_list[i] ,  'score: ' , results)


if __name__ == '__main__':
    main()
