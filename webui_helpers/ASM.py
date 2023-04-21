# Author: Théo Gieruc
# Date: 2023-01-06
# Description: Everything related to segmentation: model, dataset, visualization, etc.

import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from matplotlib.patches import Polygon
from PIL import Image
from skimage.measure import find_contours


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933], [0,0,0]]


class SegmentationModel:
    """A class that represents a segmentation model."""

    def __init__(self, path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Initializes the segmentation model, with the given path.

        Args:
            path: The path to the model file.
            device: The device to run the model on.
        """
        self.model = torch.load(path, map_location=device)
        self.model.to(device)
        self.model.eval()
        self.device = device


    @torch.no_grad()
    def inference(self, data):
        """Performs inference on the given data.

        Args:
            data (dict): A dictionary containing the data to perform inference on.
                The dictionary should have the following keys:
                - 'mask': A torch.Tensor of shape (num_masks, height, width) containing the masks for the input image.
                - 'input': A torch.Tensor of shape (num_masks, 3, height, width) containing the input image.
                - 'original_shape': A tuple of (height, width) representing the original shape of the input image.

        Returns:
            A torch.Tensor of shape (num_masks, height, width) containing the predictions for each mask.
        """

        # Return early if mask is not present
        if data['segmentation'] is None:
            return data, None

        predictions = []

        # Compute predictions for each mask
        for img in data['input']:
            prediction = self.model(img[None,:,:,:].to(self.device)).cpu().detach().numpy()[0]
            predictions.append(prediction)

        filtered_segmentation = []
        bboxes = []

        # Apply mask to prediction in order to filter out noise outside of the mask
        for mask, prediction in zip(data['segmentation'], predictions):
            filtered_segmentation.append(mask * prediction)
            bboxes.append(self.get_bbox_from_mask(mask[0]))

        filtered_segmentation = torch.cat(filtered_segmentation)

        # Resize filtered_segmentation to original image size
        if data['original_shape'][0] > data['original_shape'][1]:
            resize = T.Resize(data['original_shape'][0], interpolation=T.InterpolationMode.NEAREST)
            resized_segmentation = resize(filtered_segmentation)
            resized_segmentation = resized_segmentation[:,:,(resized_segmentation.shape[2] - data['original_shape'][1]) // 2:
                                (resized_segmentation.shape[2] - data['original_shape'][1]) // 2 + data['original_shape'][1]]
        else:
            resize = T.Resize(data['original_shape'][1], interpolation=T.InterpolationMode.NEAREST)
            resized_segmentation = resize(filtered_segmentation)
            resized_segmentation = resized_segmentation[:, (resized_segmentation.shape[1]  - data['original_shape'][0]) // 2:
                                (resized_segmentation.shape[1]  - data['original_shape'][0]) // 2 + data['original_shape'][0]]

        return resized_segmentation


    def single_inference(self, img, data, transforms=None, transforms_mask=None, desired_size=352):
        """Performs inference on the given data.

        Args:
            img: A numpy array of shape (height, width, 3) containing the image to perform inference on.
            data: A dictionary containing the data to perform inference on.
                The dictionary should have the following keys
                - 'bbox': A list of bounding boxes of shape (num_masks, 4) containing the bounding boxes for the input image.
                - 'conf': A list of confidence scores of shape (num_masks) containing the confidence scores for the input image.
            transforms: A torchvision.transforms object to apply to the input image.
            transforms_mask: A torchvision.transforms object to apply to the mask.
            desired_size: The desired size of the input image.

        Returns:
            A torch.Tensor of shape (num_masks, height, width) containing the predictions for each mask.
        """

        # Return early if mask is not present
        if len(data['scores']) == 0:
            return None       
        
        # Create default transforms if none are provided
        if transforms is None:
            transforms = T.Compose([
                                    T.ToTensor(),
                                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
        if transforms_mask is None:
            transforms_mask = T.Compose([
                                    T.ToTensor(),
                                ])

        mask = np.zeros((len(data['bounding_boxes']), img.shape[0], img.shape[1]))

        # Resize image to desired size, and pad with black pixels
        old_size = img.shape[:2]
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = cv2.resize(img, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0]
        )

        # Create mask for each bounding box
        new_mask = []

        for i, box in enumerate(data['bounding_boxes']):
            box[box < 0] = 0
            box = box.int()
            mask[i, box[1] : box[3], box[0] : box[2]] = 1
            new_mask.append(
                cv2.copyMakeBorder(
                    cv2.resize(mask[i], (new_size[1], new_size[0])),
                    top,
                    bottom,
                    left,
                    right,
                    cv2.BORDER_CONSTANT,
                    value=[0,0,0],
                )
            )

        masks = np.array(new_mask)

        # Apply transforms to image and mask
        if transforms_mask is not None:
            transformed_mask = []
            for i, mask in enumerate(masks):
                transformed_mask.append(transforms_mask(mask))
            masks = torch.stack(transformed_mask)
        if transforms is not None:
            img = transforms(img)


        # Apply mask to image 
        input_to_model = torch.zeros((len(data['bounding_boxes']), 3, img.shape[1], img.shape[2]))
        for i, mask in enumerate(masks):
            input_to_model[i] = img * mask

        # Run inference on image
        segmentation = self.inference( {'segmentation': masks, 'original_shape': old_size, 'input': input_to_model, 'data': data})

        return segmentation

    def get_bbox_from_mask(self, mask, mask_value=1):
        """Returns the bounding box of the given mask."""
        # Make sure the mask is a numpy array
        mask = np.array(mask)
        # Make sure mask values are non-negative
        mask[mask < 0] = 0
        # Get the indices of the mask that are equal to the mask value
        if mask_value is None:
            indices = np.where(mask != 0)
        else:
            indices = np.where(mask == mask_value)
        # Return a zero size box if there are no indices in the mask
        if indices[0].size <= 0 or indices[1].size <= 0:
            return np.zeros((4,), dtype=int)
        # Get the min and max values of the indices
        min_x = np.min(indices[1])
        min_y = np.min(indices[0])
        max_x = np.max(indices[1])
        max_y = np.max(indices[0])
        # Return the bounding box
        return [min_x, min_y, max_x, max_y]





def process_word(word):
    """Processes a word by removing special characters and spaces.

    Args:
        word (str): The word to process.

    Returns:
        str: The processed word.
    """
    return word.replace('.','').replace(' ', '').replace(')','').replace('(','').replace('-','').lower()

def apply_mask(image, mask, color, ax, alpha=0.3):
    """Apply the given mask to the image.

    Args:
        image (numpy.ndarray): The image to apply the mask to.
        mask (numpy.ndarray): The mask to apply to the image.
        color (list or tuple): The color to use for the mask.
        alpha (float, optional): The transparency of the mask. Default is 0.3.

    Returns:
        numpy.ndarray: The image with the applied mask.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    for verts in contours:
        verts = np.fliplr(verts) - 1
        p = Polygon(verts, facecolor="none", edgecolor=color)
        ax.add_patch(p)
    return image

def get_color(label, set_label):
    for i, elem in enumerate(set_label):
        if process_word(elem) == process_word(label):
            return i
    return 6

def color_text(color, text, label, c):
    """Color specified label in text."""
    # Find label in text and change its color
    index = [i for i, word in enumerate(text) if word.lower() == label.lower()]
    if index:
        color[index[0]] = c
    else:
        index = [i for i, word in enumerate(text) if process_word(word) == process_word(label)]
        if index:
            color[index[0]] = c
        else:
            # Split text and label into subwords
            sub_text = [word.split() for word in text]
            sub_text = [word for sublist in sub_text for word in sublist]
            sub_label = label.split()
            # Find subwords in text
            index = [i for sub_word in sub_label for i, word in enumerate(sub_text) if word.lower() == sub_word.lower()]
            # Find words containing index subwords
            final_index = [i for index_ in index for i, word in enumerate(text) if word.find(sub_text[index_]) > -1]
            if final_index:
                color[final_index[0]] = c
            else:
                print(f'No match for {label} in {text}')
    return color

def plot_results(img, data, expression='', segmentations=None, ax=None, conf=0.7):
    """Plots the results of the inference.

    Args:
        img (numpy.ndarray): The image to plot the results on.
        data (dict): The results of the inference. The dictionary should contain the following
            keys: 'bbox', 'labels', 'conf', 'scores'.
        ax (matplotlib.axes.Axes, optional): The axes to plot the results on. If None, the current
            axes will be used. Default is None.
        conf (float, optional): The confidence threshold to use for the results. Default is 0.7.

    Returns:
        dict: the text and colors of the expression
    """
    # Get current axes if none are given
    if ax is None:
        ax = plt.gca()
    # Return early if no results
    if data is None:
        return ax

    # Filter results by confidence threshold
    keep = data['scores'] > conf
    scores = data['scores'][keep]
    boxes = data['bounding_boxes'][keep]
    labels = [label for i, label in enumerate(data['labels']) if keep[i]]
    set_label = set(labels)
    if segmentations is not None:
        segmentations = segmentations[keep]
    else:
        segmentations = [None] * len(boxes)
    
    np_image = np.array(img)


    labels = [x for _,x in sorted(zip(boxes[:,0],labels))]
    scores = [x for _,x in sorted(zip(boxes[:,0],scores))]
    segmentations = [x for _,x in sorted(zip(boxes[:,0],segmentations),key=lambda x: x[0])]
    boxes = [x for _,x in sorted(zip(boxes[:,0],boxes),key=lambda x: x[0])]
    set_label = [label.replace('( ', '(').replace(' )',')') for label in set_label]

    text = [expression for expression in re.split('(' + ('|').join(set_label).replace('(','\(').replace(')','\)') + ')', expression, flags=re.IGNORECASE) if expression.strip()]
    colors = [[0,0,0] for word in text]

    for l, segmentation in zip(labels, segmentations):
        if segmentation is None:
            continue
        # find the color of the label
        c = COLORS[get_color(l, set_label)]
        # color the label in text
        colors = color_text(colors, text, l, c)
        # apply mask to image
        np_image = apply_mask(np_image, segmentation, c, ax)



    ax.imshow(np_image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    return {'text': text, 'colors': colors}


class SegmentationDataset(torch.utils.data.Dataset):
    """Segmentation dataset class, used to load images and masks from a dataframe."""
    def __init__(
        self,
        df_dataset,
        transforms=None,
        transforms_mask=None,
        images_folder=None,
        desired_size=352
    ):
        self.transforms = transforms
        self.transforms_mask = transforms_mask
        self.df_dataset = df_dataset
        self.images_folder = images_folder
        self.desired_size = desired_size

    def __getitem__(self, idx):
        data = self.df_dataset.loc[idx].copy()
        img_path = self.images_folder + data['filename']
        pilimg = Image.open(img_path).convert("RGB")
        img = np.array(pilimg)
        mask = np.zeros((len(data['bounding_boxes']), img.shape[0], img.shape[1]))

        old_size = img.shape[:2]
        ratio = float(self.desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = cv2.resize(img, (new_size[1], new_size[0]))

        delta_w = self.desired_size - new_size[1]
        delta_h = self.desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        new_mask = []
        if len(data['conf']) == 0:
            return {
            'original_shape': old_size,
            'input': None,
            'segmentation': None,
            'data': data,
            }
        for i, box in enumerate(data['bounding_boxes']):
            box[box < 0] = 0
            box = box.int()
            mask[i, box[1] : box[3], box[0] : box[2]] = 1
            new_mask.append(
                cv2.copyMakeBorder(
                    cv2.resize(mask[i], (new_size[1], new_size[0])),
                    top,
                    bottom,
                    left,
                    right,
                    cv2.BORDER_CONSTANT,
                    value=color,
                )
            )
        mask = np.array(new_mask)

        if self.transforms_mask is not None:
            transformed_mask = []
            for i, mask_ in enumerate(mask):
                transformed_mask.append(self.transforms_mask(mask_))
            mask = torch.stack(transformed_mask)
        if self.transforms is not None:
            img = self.transforms(img)

        filtered_segmentation = torch.zeros((len(data['bbox']), 3, img.shape[1], img.shape[2]))
        for i, mask_ in enumerate(mask):
            filtered_segmentation[i] = img * mask_

        return {
            'original_shape': old_size,
            'input': filtered_segmentation,
            'segmentation': mask,
            'data': data,
        }

    def __len__(self):
        return len(self.df_dataset)

def colored_text_to_html(colored_text):
    # Get the text and color data for the image
    text = colored_text["text"]
    colors = colored_text["colors"]

    # Create an empty list to store the HTML text
    html_text = []

    # Iterate over the text and color data
    for i in range(len(text)):
        # Convert the RGB color values to hexadecimal
        color = "#{:02x}{:02x}{:02x}".format(
            int(colors[i][0] * 255),
            int(colors[i][1] * 255),
            int(colors[i][2] * 255)
        )

        # Add the text and color to the HTML text list
        html_text.append(f'<span style="color: {color}">{text[i]}</span>')

    # Join the HTML text list into a single string
    html_text = "".join(html_text)
    return html_text