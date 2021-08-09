# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:10:33 2020

@author: NAT
"""
import random
import warnings

import PIL
import torch
# import torchvision.transforms.functional as F
import torch.nn.functional as F
warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================BEGIN CACULATE IoU==================================
def intersect(boxes1, boxes2):
    '''
        Find intersection of every box combination between two sets of box
        boxes1: bounding boxes 1, a tensor of dimensions (n1, 4)
        boxes2: bounding boxes 2, a tensor of dimensions (n2, 4)

        Out: Intersection each of boxes1 with respect to each of boxes2,
             a tensor of dimensions (n1, n2)
    '''
    n1 = boxes1.size(0)
    n2 = boxes2.size(0)
    max_xy = torch.min(boxes1[:, 2:].unsqueeze(1).expand(n1, n2, 2),
                       boxes2[:, 2:].unsqueeze(0).expand(n1, n2, 2))

    min_xy = torch.max(boxes1[:, :2].unsqueeze(1).expand(n1, n2, 2),
                       boxes2[:, :2].unsqueeze(0).expand(n1, n2, 2))
    inter = torch.clamp(max_xy - min_xy, min=0)  # (n1, n2, 2)
    return inter[:, :, 0] * inter[:, :, 1]  # (n1, n2)


def find_IoU(boxes1, boxes2):
    '''
        Find IoU between every boxes set of boxes
        boxes1: a tensor of dimensions (n1, 4) (left, top, right , bottom)
        boxes2: a tensor of dimensions (n2, 4)

        Out: IoU each of boxes1 with respect to each of boxes2, a tensor of
             dimensions (n1, n2)

        Formula:
        (box1 ∩ box2) / (box1 u box2) = (box1 ∩ box2) / (area(box1) + area(box2) - (box1 ∩ box2 ))
    '''
    inter = intersect(boxes1, boxes2)
    area_boxes1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area_boxes2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    area_boxes1 = area_boxes1.unsqueeze(1).expand_as(inter)  # (n1, n2)
    area_boxes2 = area_boxes2.unsqueeze(0).expand_as(inter)  # (n1, n2)
    union = (area_boxes1 + area_boxes2 - inter)
    return inter / union


# ==========================END CACULATE IoU====================================

# ==========================BEGIN AUGMENTATION==================================
# Distort
def distort(image):
    '''
    Distort brightness, contrast, saturation
    image: A PIL image

    Out: New image (PIL)
    '''

    if type(image) != PIL.Image.Image:
        image = F.to_pil_image(image)
    new_image = image
    distortions = [F.adjust_brightness,
                   F.adjust_contrast,
                   F.adjust_saturation]

    random.shuffle(distortions)

    for function in distortions:
        if random.random() < 0.5:
            adjust_factor = random.uniform(0.5, 1.5)
            new_image = function(new_image, adjust_factor)

    return new_image


# -----------------------------------------------------------
# lighting_noise
def lighting_noise(image):
    '''
        color channel swap in image
        image: A PIL image

        Out: New image with swap channel (Probability = 0.5, PIL image)
    '''
    if type(image) != PIL.Image.Image:
        image = F.to_pil_image(image)
    new_image = image
    if random.random() < 0.5:
        perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2),
                 (1, 2, 0), (2, 0, 1), (2, 1, 0))
        swap = perms[random.randint(0, len(perms) - 1)]
        new_image = F.to_tensor(new_image)
        new_image = new_image[swap, :, :]
        new_image = F.to_pil_image(new_image)
    return new_image


# -----------------------------------------------------------
# Resize
def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
    '''
        Resize image to (300, 300)  for SSD300
        image: A PIL image
        boxes: bounding boxes, a tensor of dimensions (n_objects, 4)

        Out:New image, new boxes or percent coordinates
    '''
    if type(image) != PIL.Image.Image:
        image = F.to_pil_image(image)
    new_image = F.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


# -----------------------------------------------------------
# Expand with filler
def expand_filler(image, boxes, filler):
    '''
        Perform a zooming out operation by placing the
        image in a larger canvas of filler material. Helps to learn to detect
        smaller objects.
        image: input image, a tensor of dimensions (3, original_h, original_w)
        boxes: bounding boxes, a tensor of dimensions (n_objects, 4)
        filler: RBG values of the filler material, a list like [R, G, B]

        Out: new_image (A Tensor), new_boxes
    '''
    if type(image) == PIL.Image.Image:
        image = F.to_tensor(image)
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create an image with the filler
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)

    # Place the original image at random coordinates
    # in this new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h

    new_image[:, top:bottom, left:right] = image

    # Adjust bounding box
    boxes = torch.tensor(boxes)
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0)

    return new_image, new_boxes


# -----------------------------------------------------------
# Random crop
def random_crop(image, boxes, labels, difficulties):
    '''
        Performs a random crop. Helps to learn to detect larger and partial object
        image: A tensor of dimensions (3, original_h, original_w)
        boxes: Bounding boxes, a tensor of dimensions (n_objects, 4)
        labels: labels of object, a tensor of dimensions (n_objects)
        difficulties: difficulties of detect object, a tensor of dimensions (n_objects)

        Out: cropped image (Tensor), new boxes, new labels, new difficulties
    '''
    if type(image) == PIL.Image.Image:
        image = F.to_tensor(image)
    original_h = image.size(1)
    original_w = image.size(2)

    while True:
        mode = random.choice([0.1, 0.3, 0.5, 0.9, None])

        if mode is None:
            return image, boxes, labels, difficulties

        new_image = image
        new_boxes = boxes
        new_difficulties = difficulties
        new_labels = labels
        for _ in range(50):
            # Crop dimensions: [0.3, 1] of original dimensions
            new_h = random.uniform(0.3 * original_h, original_h)
            new_w = random.uniform(0.3 * original_w, original_w)

            # Aspect ratio constraint b/t .5 & 2
            if new_h / new_w < 0.5 or new_h / new_w > 2:
                continue

            # Crop coordinate
            left = random.uniform(0, original_w - new_w)
            right = left + new_w
            top = random.uniform(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([int(left), int(top), int(right), int(bottom)])

            # Calculate IoU  between the crop and the bounding boxes
            overlap = find_IoU(crop.unsqueeze(0), boxes)  # (1, n_objects)
            overlap = overlap.squeeze(0)
            # If not a single bounding box has a IoU of greater than the minimum, try again
            if overlap.max().item() < mode:
                continue

            # Crop
            new_image = image[:, int(top):int(bottom), int(left):int(right)]  # (3, new_h, new_w)

            # Center of bounding boxes
            center_bb = (boxes[:, :2] + boxes[:, 2:]) / 2.0

            # Find bounding box has been had center in crop
            center_in_crop = (center_bb[:, 0] > left) * (center_bb[:, 0] < right
                                                         ) * (center_bb[:, 1] > top) * (
                                     center_bb[:, 1] < bottom)  # (n_objects)

            if not center_in_crop.any():
                continue

            # take matching bounding box
            new_boxes = boxes[center_in_crop, :]

            # take matching labels
            new_labels = labels[center_in_crop]

            # take matching difficulities
            new_difficulties = difficulties[center_in_crop]

            # Use the box left and top corner or the crop's
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])

            # adjust to crop
            new_boxes[:, :2] -= crop[:2]

            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])

            # adjust to crop
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficulties

        return new_image, new_boxes, new_labels, new_difficulties
    # -----------------------------------------------------------


# random flip
def random_flip(image, boxes):
    '''
        Flip image horizontally.
        image: a PIL image
        boxes: Bounding boxes, a tensor of dimensions (n_objects, 4)

        Out: flipped image (A PIL image), new boxes
    '''
    if type(image) != PIL.Image.Image:
        image = F.to_pil_image(image)
    if random.random() > 0.5:
        return image, boxes
    new_image = F.hflip(image)

    # flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0]
    new_boxes[:, 2] = image.width - boxes[:, 2]
    new_boxes = new_boxes[:, [2, 1, 0, 3]]
    return new_image, new_boxes


# -----------------------------------------------------------
# Transform
def transform_od(image, boxes, labels, difficulties=0, split="TRAIN"):
    '''
        Apply transformation
        image: A PIL image
        boxes: bounding boxe, a tensor of dimensions (n_objects, 4)
        labels: labels of object a tensor of dimensions (n_object)
        difficulties: difficulties of object detect, a tensor of dimensions (n_object)
        split: one of "TRAIN", "TEST"

        Out: transformed images, transformed bounding boxes, transformed labels,
        transformed difficulties
    '''

    if type(image) != PIL.Image.Image:
        image = F.to_pil_image(image)
    split = split.upper()
    if split not in {"TRAIN", "TEST"}:
        print("Param split in transform not in {TRAIN, TEST}")
        assert split in {"TRAIN", "TEST"}

    # mean and std from ImageNet
    mean = [0.4690646, 0.4407227, 0.40508908]
    std = [0.2514227, 0.24312855, 0.24266963]

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    # Skip transform for testing
    if split == "TRAIN":
        # Apply distort image
        new_image = distort(new_image)

        # Apply lighting noise
        new_image = lighting_noise(new_image)
        # Expand image
        if random.random() < 0.5:
            new_image, new_boxes = expand_filler(new_image, boxes, mean)

        # # Random crop
        # new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image,
        #                                                                  new_boxes,
        #                                                                  new_labels, new_difficulties)

        # Flip image
        new_image, new_boxes = random_flip(new_image, new_boxes)

    # Resize image to (300, 300)
    new_image, new_boxes = resize(new_image, new_boxes, dims=(256, 256))

    new_image = F.to_tensor(new_image)
    new_image = F.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels, new_difficulties


# ==========================END AUGMENTATION====================================
def combine(batch):
    '''
        Combine these tensors of different sizes in batch.
        batch: an iterable of N sets from __getitem__()
    '''
    images = []
    boxes = []
    labels = []
    difficulties = []

    for b in batch:
        images.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])
        difficulties.append(b[3])

    images = torch.stack(images, dim=0)
    return images, boxes, labels, difficulties


def decimate(tensor, m):
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor


# =====================BEGIN CONVERT BBOXES=======================================
def xy_to_cxcy(bboxes):
    '''
        Convert bboxes from (xmin, ymin, xmax, ymax) to (cx, cy, w, h)
        bboxes: Bounding boxes, a tensor of dimensions (n_object, 4)

        Out: bboxes in center coordinate
    '''
    return torch.cat([(bboxes[:, 2:] + bboxes[:, :2]) / 2,
                      bboxes[:, 2:] - bboxes[:, :2]], 1)


def cxcy_to_xy(bboxes):
    '''
        Convert bboxes from (cx, cy, w, h) to (xmin, ymin, xmax, ymax)
    '''
    return torch.cat([bboxes[:, :2] - (bboxes[:, 2:] / 2),
                      bboxes[:, :2] + (bboxes[:, 2:] / 2)], 1)


def encode_bboxes(bboxes, default_boxes):
    '''
        Encode bboxes correspoding default boxes (center form)

        Out: Encodeed bboxes to 4 offset, a tensor of dimensions (n_defaultboxes, 4)
    '''
    return torch.cat([(bboxes[:, :2] - default_boxes[:, :2]) / (default_boxes[:, 2:] / 10),
                      torch.log(bboxes[:, 2:] / default_boxes[:, 2:]) * 5], 1)


def decode_bboxes(offsets, default_boxes):
    '''
        Decode offsets
    '''
    return torch.cat([offsets[:, :2] * default_boxes[:, 2:] / 10 + default_boxes[:, :2],
                      torch.exp(offsets[:, 2:] / 5) * default_boxes[:, 2:]], 1)
