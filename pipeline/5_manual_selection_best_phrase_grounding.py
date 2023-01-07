import copy
import pickle
import sys
from itertools import compress
from pathlib import Path
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib.patches import Polygon
from skimage.measure import find_contours
from torchvision.ops import nms
from os.path import join
import src.GLIP.maskrcnn_benchmark as maskrcnn_benchmark

sys.modules['maskrcnn_benchmark'] = maskrcnn_benchmark

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, help="Path to the input file")
parser.add_argument("--output_file", type=str, help="Path to the output file")
parser.add_argument("--selection_column", type=str, help="Column to store the selection")
parser.add_argument("--from_row", type=int, default=0, help="Row to start from")
parser.add_argument("--image_directory", type=str, help="Path to the image directory")
args = parser.parse_args()

i = 0
cid = 0
fig, ax = plt.subplots(2, 2, figsize=(35, 20))


# Load the data
with open(args.input_file, 'rb') as f:
    data = pickle.load(f)

data[args.selection_column] = [None for _ in range(len(data))]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def get_color(label, set_label):
    for i, elem in enumerate(set_label):
        if elem == label:
            return i
    return 0


def plot_results(ax, pil_img, results, masks=None, conf=0.7):
    if len(results) == 0:
        return ax
    np_image = np.array(pil_img)
    colors = COLORS * 100
    keep = results[0] > conf
    scores = results[0][keep]
    boxes = results[1][keep]
    labels = list(compress(results[2], keep))
    set_label = set(labels)

    if masks is None:
        masks = [None for _ in range(len(scores))]
    else:
        masks = masks[results[0] > conf]

    for s, (xmin, ymin, xmax, ymax), l, mask in zip(scores, boxes.tolist(), labels, masks):
        c = colors[get_color(l, set_label)]
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=1))
        text = f'{l}: {s:0.2f}'
        ax.text(int(xmin), int(ymin), text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        if mask is None:
            continue
        np_image = apply_mask(np_image, mask, c)

        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=c)
            ax.add_patch(p)

    ax.imshow(np_image)

    return ax



def save_result(i, model, expr):
    data.loc[i, args.selection_column] = model + ' ' + expr

def on_press(event):
    global i
    global ax
    
    for ax_ in ax:
        for ax__ in ax_:
            ax__.clear()

    # QUIT
    if event.key == 'x':
        plt.close('all')
        with open('temp/latest.txt', 'w') as f:
            f.write(i)
        print(f'Latest: {i}')
    # MDETR CAPTION
    elif event.key == '7':
        # plt.close('all')
        save_result(i, 'MDETR', 'caption')
        i += 1
        load_slide()
    # GLIP CAPTION
    elif event.key == '9':
        # plt.close('all')
        save_result(i, 'GLIP', 'caption')
        i += 1
        load_slide()
    # MDETR TITLE
    elif event.key == '1':
        # plt.close('all')
        save_result(i, 'MDETR', 'title')
        i += 1
        load_slide()
    # GLIP TITLE
    elif event.key == '3':
        # plt.close('all')
        save_result(i, 'GLIP', 'title')
        i += 1
        load_slide()
    # PASS
    elif event.key == 'p':
        # plt.close('all')
        with open('temp/none.txt', 'a') as file:
            file.write(str(i) + ' ')
        i += 1
        load_slide()
    elif event.key == 'r':
        i -= 1
        load_slide()


def load_slide():
    global i
    global fig, ax
    row = data.iloc[i]
    print(i)
    no_results = False
    for column in ['MDETR_caption', 'MDETR_title', 'GLIP_caption', 'GLIP_title']:
        if row[column] is None or len(row[column]) == 0:
            no_results = True
    if no_results:
        i += 1
        load_slide()
        return


    # fig.suptitle(f'{elem["caption"]["raw"]}\n{elem["title"]["raw"]}', fontsize=15)
    im = Image.open(join(args.image_directory, row['filename'])).convert('RGB')

    ax[0, 0] = plot_results(ax=ax[0, 0], pil_img=im, results=row['MDETR_caption'])
    ax[0, 0].title.set_text(f'MDETR {row["caption"]}')

    ax[1, 0] = plot_results(ax=ax[1, 0], pil_img=im, results=row['MDETR_title'])
    ax[1, 0].title.set_text(f'MDETR {row["title"]}')

    ax[0, 1] = plot_results(ax=ax[0, 1], pil_img=im, results=row['GLIP_caption'], conf=0)
    ax[0, 1].title.set_text(f'GLIP {row["caption"]}')

    ax[1, 1] = plot_results(ax=ax[1, 1], pil_img=im, results=row['GLIP_title'], conf=0)
    ax[1, 1].title.set_text(f'GLIP {row["title"]}')

    for ax_ in ax:
        for ax__ in ax_:
            ax__.axis('off')
    # plt.savefig(f'./figs_mdetr_glip/{i}.png')
    # plt.close()
    plt.draw()

cid = fig.canvas.mpl_connect('key_press_event', on_press)

plt.show()
load_slide()

# save data as pickle
with open('args.output_file', 'wb') as f:
    pickle.dump(data, f)