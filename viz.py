from PIL import Image
from got10k.datasets import GOT10k
from got10k.utils.viz import show_frame
import numpy as np
import cv2
import os

def draw_img(img, bbox, idx=0, rank=1):
    pred_bbox = gen_pos(bbox)

    if rank==0:
        color = (0, 0, 255)
    elif rank == 1:
        color = (0, 255, 0)
    elif rank == 2:
        color = (0, 255, 255)
    else:
        color = (255, 0, 0)

    img = cv2.rectangle(img, pred_bbox[0], pred_bbox[1], color, 6)
    img = cv2.putText(img, str(idx), (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 4)

    return img

def gen_pos(bbox):
    min_x, min_y, w, h = bbox
    x0 = np.round(min_x).astype(int)
    y0 = np.round(min_y + h).astype(int)
    x1 = np.round(min_x + w).astype(int)
    y1 = np.round(min_y).astype(int)
    pos0, pos1 = (x0, y0), (x1, y1)

    return pos0, pos1

dataset = GOT10k(root_dir='data/GOT-10k', subset='test')
trackers = ['Ours+AC+DKLm6', 'SiamRPN']
# indexing
img_file, _ = dataset[10]

# for-loop
for s, (img_files, anno) in enumerate(dataset):
    results = []
    seq_name = dataset.seq_names[s]
    print('Sequence:', seq_name)

    for t in range(len(trackers)):
        anno_path = "./results/GOT-10k/{0}/{1}/{2}_001.txt".format(trackers[t], seq_name, seq_name)
        results.append(np.loadtxt(anno_path, delimiter=',', dtype=float))
    # show all frames
    for f, img_file in enumerate(img_files):
        image = cv2.imread(img_file)
        for t in range(len(trackers)):
            image = draw_img(image, results[t][f], idx=f, rank=t)
        image = cv2.resize(image, (512, 512))
        if not os.path.exists("./temp/got10k/imgs/{0}/".format(seq_name)):
            os.mkdir("./temp/got10k/imgs/{0}/".format(seq_name))
        cv2.imwrite("./temp/got10k/imgs/{0}/{1}_{2}.jpg".format(seq_name, seq_name, f), image)
        # cv2.imshow("Viz", image)
        # cv2.waitKey(1)
