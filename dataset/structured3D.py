#!/usr/bin/env python
"""Process Gillsjo's Structured3D-SRW dataset for L-CNN network
Usage:
    dataset/structured3D.py <src> <dst>
    dataset/structured3D.py (-h | --help )

Examples:
    python dataset/structured3D.py /datadir/wireframe data/wireframe

Arguments:
    <src>                Original data directory of Gillsjo's Structured3D-SRW dataset
    <dst>                Directory of the output

Options:
   -h --help             Show this screen.
"""

import os
import sys
import json
from itertools import combinations

import cv2
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt
from docopt import docopt
from scipy.ndimage import zoom
from tqdm import tqdm

try:
    sys.path.append(".")
    sys.path.append("..")
    from lcnn.utils import parmap
except Exception:
    raise


def inrange(v, shape):
    return 0 <= v[0] < shape[0] and 0 <= v[1] < shape[1]


def to_int(x):
    return tuple(map(int, x))


def save_heatmap(prefix, image, junctions, edges_pos, edges_neg):
    im_rescale = (512, 512)
    heatmap_scale = (128, 128)

    fy, fx = heatmap_scale[1] / image.shape[0], heatmap_scale[0] / image.shape[1]
    jmap = np.zeros((1,) + heatmap_scale, dtype=np.float32)
    joff = np.zeros((1, 2) + heatmap_scale, dtype=np.float32)
    lmap = np.zeros(heatmap_scale, dtype=np.float32)

    junctions[:, 0] = np.clip(junctions[:, 0] * fx, 0, heatmap_scale[0] - 1e-4)
    junctions[:, 1] = np.clip(junctions[:, 1] * fy, 0, heatmap_scale[1] - 1e-4)
    junc = np.hstack([junctions[:,::-1], np.zeros([junctions.shape[0],1])]).astype(np.float32)
    Lpos = edges_pos.astype(np.int)
    lpos = np.stack([
        junc[edges_pos[:,0]],
        junc[edges_pos[:,1]]
        ], axis=1)
    Lneg = edges_neg.astype(np.int)
    lneg = np.stack([
        junc[edges_neg[:,0]],
        junc[edges_neg[:,1]]
        ], axis=1)



    yint,xint = to_int(junc[:,0]), to_int(junc[:,1])
    off_x = junc[:,1] - xint -0.5
    off_y = junc[:,0] - yint -0.5
    joff[0,0,yint,xint] = off_y
    joff[0,1,yint,xint] = off_x
    jmap[0,yint,xint] = 1


    for l_edge in edges_pos:
        e1,e2 = l_edge
        rr, cc, value = skimage.draw.line_aa(yint[e1],xint[e1], yint[e2],xint[e2])
        lmap[rr, cc] = np.maximum(lmap[rr, cc], value)

    assert len(lneg) != 0

    image = cv2.resize(image, im_rescale)

    # plt.figure()
    # plt.subplot(131), plt.imshow(lmap)
    # plt.subplot(132), plt.imshow(image)
    # for i0, i1 in Lpos:
    #     plt.scatter(junc[i0][1] * 4, junc[i0][0] * 4)
    #     plt.scatter(junc[i1][1] * 4, junc[i1][0] * 4)
    #     plt.plot([junc[i0][1] * 4, junc[i1][1] * 4], [junc[i0][0] * 4, junc[i1][0] * 4])
    # plt.subplot(133), plt.imshow(lmap)
    # for i0, i1 in Lneg[:150]:
    #     plt.plot([junc[i0][1], junc[i1][1]], [junc[i0][0], junc[i1][0]])
    # plt.savefig(f"{prefix}_plot.png",)
    # plt.close()

    # For junc, lpos, and lneg that stores the junction coordinates, the last
    # dimension is (y, x, t), where t represents the type of that junction.  In
    # the wireframe dataset, t is always zero.
    np.savez_compressed(
        f"{prefix}_label.npz",
        aspect_ratio=image.shape[1] / image.shape[0],
        jmap=jmap,  # [J, H, W]    Junction heat map
        joff=joff,  # [J, 2, H, W] Junction offset within each pixel
        lmap=lmap,  # [H, W]       Line heat map with anti-aliasing
        junc=junc,  # [Na, 3]      Junction coordinate
        Lpos=Lpos,  # [M, 2]       Positive lines represented with junction indices
        Lneg=Lneg,  # [M, 2]       Negative lines represented with junction indices
        lpos=lpos,  # [Np, 2, 3]   Positive lines represented with junction coordinates
        lneg=lneg,  # [Nn, 2, 3]   Negative lines represented with junction coordinates
    )
    cv2.imwrite(f"{prefix}.png", image)

    # plt.imshow(jmap[0])
    # plt.savefig("/tmp/1jmap0.jpg")
    # plt.imshow(jmap[1])
    # plt.savefig("/tmp/2jmap1.jpg")
    # plt.imshow(lmap)
    # plt.savefig("/tmp/3lmap.jpg")
    # plt.imshow(Lmap[2])
    # plt.savefig("/tmp/4ymap.jpg")
    # plt.imshow(jwgt[0])
    # plt.savefig("/tmp/5jwgt.jpg")
    # plt.cla()
    # plt.imshow(jmap[0])
    # for i in range(8):
    #     plt.quiver(
    #         8 * jmap[0] * cdir[i] * np.cos(2 * math.pi / 16 * i),
    #         8 * jmap[0] * cdir[i] * np.sin(2 * math.pi / 16 * i),
    #         units="xy",
    #         angles="xy",
    #         scale_units="xy",
    #         scale=1,
    #         minlength=0.01,
    #         width=0.1,
    #         zorder=10,
    #         color="w",
    #     )
    # plt.savefig("/tmp/6cdir.jpg")
    # plt.cla()
    # plt.imshow(lmap)
    # plt.quiver(
    #     2 * lmap * np.cos(ldir),
    #     2 * lmap * np.sin(ldir),
    #     units="xy",
    #     angles="xy",
    #     scale_units="xy",
    #     scale=1,
    #     minlength=0.01,
    #     width=0.1,
    #     zorder=10,
    #     color="w",
    # )
    # plt.savefig("/tmp/7ldir.jpg")
    # plt.cla()
    # plt.imshow(jmap[1])
    # plt.quiver(
    #     8 * jmap[1] * np.cos(tdir),
    #     8 * jmap[1] * np.sin(tdir),
    #     units="xy",
    #     angles="xy",
    #     scale_units="xy",
    #     scale=1,
    #     minlength=0.01,
    #     width=0.1,
    #     zorder=10,
    #     color="w",
    # )
    # plt.savefig("/tmp/8tdir.jpg")


def main():
    args = docopt(__doc__)
    data_root = args["<src>"]
    data_output = args["<dst>"]

    os.makedirs(data_output, exist_ok=True)
    for batch in ["train", "val", "test"]:
        print(f'Batch {batch}')
        anno_file = os.path.join(data_root, f"{batch}.json")

        with open(anno_file, "r") as f:
            dataset = json.load(f)

        def handle(data):
            im = cv2.imread(os.path.join(data_root, "images", data["filename"]))
            prefix = data["filename"].split(".")[0]
            junctions = np.array(data["junctions"]).reshape(-1, 2)
            edges_pos = np.array(data["edges_positive"]).reshape(-1, 2)
            edges_neg = np.array(data["edges_negative"]).reshape(-1, 2)
            if len(edges_neg) == 0:
                # print("Skip", os.path.join(data_output, batch, prefix))
                return
            os.makedirs(os.path.join(data_output, batch), exist_ok=True)


            path = os.path.join(data_output, batch, prefix)
            save_heatmap(f"{path}_0", im[::, ::], junctions, edges_pos, edges_neg)
            # print("Finishing", os.path.join(data_output, batch, prefix))

        parmap(handle, dataset, 12, progress_bar=tqdm)


if __name__ == "__main__":
    main()
