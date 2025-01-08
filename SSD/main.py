import os
import xml.etree.ElementTree as ET
import torch

from torch.utils.data import Dataset, DataLoader, random_split

from torchvision.models.detection import ssd300_vgg16
import torchvision.transforms.v2 as transforms
from torchvision import tv_tensors
from torchvision.io import decode_image

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import torch.nn.functional as F

from torchmetrics.detection.mean_ap import MeanAveragePrecision
import json

from copy import deepcopy


class CustomDataset(Dataset):

    def __init__(self, root, train_transforms=None, test_transform=None, classes=None, apply_train_transform=True):

        self.root             = root
        self.train_transforms = train_transforms
        self.test_transform   = test_transform
        self.images_dir       = os.path.join(root, 'images')
        self.annotations_dir  = os.path.join(root, 'annotations')
        self.apply_train_transform = apply_train_transform

        self.annotations = [f.split('.')[0] for f in os.listdir(self.annotations_dir) if f.endswith('.xml')]

        self.classes = {item: index for index, item in enumerate(classes, 1)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        annotation      = self.annotations[index]
        annotation_path = os.path.join(self.annotations_dir, f"{annotation}.xml")
        image_name, boxes, labels = self.parse_voc_xml(annotation_path)

        img_path = os.path.join(self.images_dir, image_name)
        img      = decode_image(img_path, mode="RGB")

        C, H, W = img.size()

        boxes  = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        tv_boxes = tv_tensors.BoundingBoxes(
            boxes,
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(H, W)
        )

        target = {"boxes": tv_boxes, "labels": labels}

        if self.apply_train_transform and self.train_transforms:
            img, target = self.train_transforms(img, target)
        elif self.test_transform:
            img, target = self.test_transform(img, target)

        return img, target

    def parse_voc_xml(self, xml_file):

        tree = ET.parse(xml_file)
        root = tree.getroot()

        image_name = root.find("filename").text

        boxes  = []
        labels = []

        for obj in root.findall("object"):

            label = obj.find("name").text
            if label not in self.classes:
                continue

            label_idx = self.classes[label]

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_idx)

        return image_name, boxes, labels

def collate(x):
    return tuple(zip(*x))

def batch_to_device(img_batch, target_batch, device):
    return [img.to(device) for img in img_batch], [{k: v.to(device) for k, v in target.items()} for target in target_batch]



DATASET_ROOT = "./dataset"
CLASSES = [
    "with_mask",
    "without_mask",
    "mask_weared_incorrect"
]

if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_transform = transforms.Compose([
        transforms.ToImage(),

        transforms.ToDtype(torch.uint8, scale=True),

        transforms.RandomPhotometricDistort(),
        transforms.RandomRotation(degrees=10),
        
        transforms.RandomChoice([
            transforms.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
            transforms.RandomIoUCrop(),
        ]),
        transforms.RandomHorizontalFlip(),
        
        transforms.ClampBoundingBoxes(),
        transforms.SanitizeBoundingBoxes(),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    test_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    dataset = CustomDataset(
        root=DATASET_ROOT,
        classes=CLASSES,
        train_transforms=train_transform,
        test_transform=test_transform,
        apply_train_transform=True
    )
    train_set, val_set = random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers = True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=4,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers = True
    )

    model = ssd300_vgg16(
        num_classes=4,
        trainable_backbone_layers=3
    )
    model.to(device)

    best = None
    best_train_map = -1
    best_valid_map = -1

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,            
        weight_decay=1e-4,
        eps=1e-8            
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=20,
        T_mult=2,
        eta_min=1e-5
    )

    ALPHA = 0.35
    GRAD_CLIP = 2.0

    metric = MeanAveragePrecision()
    metric.warn_on_many_detections = False

    train_results = []
    valid_results = []
    track = [
        "map",
        "map_50",
        "map_75",
        "mar_100"
    ]

    model.train()
    epochs = 170
    iterations = len(train_loader)
    for epoch in range(epochs):

        print(f"Epoch: {epoch}")

        for i, (img_batch, target_batch) in enumerate(train_loader):

            new_img_batch, new_target_batch = batch_to_device(img_batch, target_batch, device=device)

            losses = model(new_img_batch, new_target_batch)

            class_loss = losses["classification"]
            regrs_loss = losses["bbox_regression"]

            loss = class_loss + ALPHA * regrs_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRAD_CLIP,
                error_if_nonfinite=True
            )

            optimizer.step()
            scheduler.step(epoch + i / iterations)

        if (epoch + 1) % 5 == 0:
            model.eval()
            dataset.apply_train_transform = False
            with torch.no_grad():

                for i, (img_batch, target_batch) in enumerate(train_loader):
                    new_img_batch, new_target_batch = batch_to_device(img_batch, target_batch, device=device)

                    preds = model(new_img_batch)

                    metric.update(
                        preds,
                        new_target_batch
                    )

                result = metric.compute()
                train_map = result["map"]
                print(result["map"])

                train_results.append(
                    ({k: v.item() for k, v in result.items() if k in track}, epoch)
                )
                metric.reset()            

                for i, (img_batch, target_batch) in enumerate(val_loader):
                    new_img_batch, new_target_batch = batch_to_device(img_batch, target_batch, device=device)

                    preds = model(new_img_batch)

                    metric.update(
                        preds,
                        new_target_batch
                    )

                result = metric.compute()
                valid_map = result["map"]
                print(result["map"])

                valid_results.append(
                    ({k: v.item() for k, v in result.items() if k in track}, epoch)
                )
                metric.reset()

                if (len(train_results) > 1) and (train_map > best_train_map) and (valid_map > best_valid_map):
                    print("New Best")
                    best = deepcopy(model)
                    best_train_map = train_map
                    best_valid_map = valid_map
            model.train()
            dataset.apply_train_transform = True

    model_scripted = torch.jit.script(model)
    model_scripted.save('model.pt')

    if best != None:
        best_scripted = torch.jit.script(best)
        best_scripted.save('best.pt')

    with open('train_res.json', 'w') as file:
        json.dump(train_results, file, indent=4)
    
    with open('valid_res.json', 'w') as file:
        json.dump(valid_results, file, indent=4)