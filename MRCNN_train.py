import matplotlib.pyplot as plt
from torchvision.io import read_image
import cv2
import torch 
import os
import torchvision
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as FTV
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.transforms import v2 as T
import utils
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import numpy as np
from engine import train_one_epoch, evaluate
import matplotlib.patches as patches
from torch import nn
from inference import inference


def tif_to_CHW(filepath):
    """Reads in a .tif file as a color image in the cv2 hwc format and converts to 
    the torch chw format"""

    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return image

class MPFDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(self.root, 'IMGs'))))
        self.masks = list(sorted(os.listdir(os.path.join(self.root, 'masks'))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(os.path.join(self.root, 'IMGs'), self.imgs[idx])
        mask_path = os.path.join(os.path.join(self.root, 'masks'), self.masks[idx])
        

        img = read_image(img_path)
        
        mask = read_image(mask_path)
        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class
        # TODO: what if we eventually have more than 1 class?
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=FTV.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes, trainable_backbone_layers=0, new_anchors = True, large_mask_head = True):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights = "DEFAULT",
                                                                progress=True, 
                                                                weights_backbone = "ResNet50_Weights.DEFAULT",
                                                                trainable_backbone_layers=trainable_backbone_layers,
                                                                rpn_batch_size_per_image= 64,
                                                                rpn_post_nms_top_n_train= 1000,
                                                                rpn_post_nms_top_n_test= 2000,
                                                                rpn_nms_thresh= 0.9,
                                                                box_batch_size_per_image=128,
                                                                min_size=512,
                                                                max_size=512,
                                                                box_detections_per_img= 150,
                                                                )
    #NOTE: SECOND TRAINING SESSION USING _default_anchorgen()
    if new_anchors:
        anchor_generator = AnchorGenerator(
        sizes=tuple([(8, 16, 32, 64, 128, 256) for _ in range(5)]),
        aspect_ratios=tuple([(0.5, 1.0, 2.0) for _ in range(5)]))
        model.rpn.anchor_generator = anchor_generator
        # 256 because that's the number of features that ResNet50_FPN returns
        model.rpn.head = torchvision.models.detection.faster_rcnn.RPNHead(256, anchor_generator.num_anchors_per_location()[0], 3)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        2
    ) 

    if large_mask_head:
        model.roi_heads.mask_head = MaskRCNNHeads(256, (256,512,1024,512,256), 1, norm_layer = nn.BatchNorm2d)

    # torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights
    # backbone = torchvision.models.resnet50(weights = 'ResNet50_Weights.DEFAULT')
    # backbone.out_channels = 256

    # anchor_generator = AnchorGenerator(sizes=((8, 16, 32, 128, 256),),
    #                                    aspect_ratios=((0.5, 1.0, 2.0),))
    
    # model = torchvision.models.detection.MaskRCNN(
    #      backbone= backbone,
    #      num_classes= 2,
    #      rpn_anchor_generator=anchor_generator,
    #      rpn_batch_size_per_image= 64,
    #      rpn_post_nms_top_n_train= 1000,
    #      rpn_post_nms_top_n_test= 2000,
    #      rpn_nms_thresh= 0.9,
    #      box_batch_size_per_image=128,
    #      min_size=512,
    #      max_size=512,
    #      box_detections_per_img= 100
    #                                               )
    return model

def get_transform(train):
    transforms = []
    # if train:
        # transforms.append(T.GaussianBlur())
        # transforms.append(T.RandomVerticalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the directory containing the dataset within 'train' and 'test' subdirectories.")
    parser.add_argument("--log_dir", type = str, required=True)
    args = parser.parse_args()


    def train_model(model, train_data_loader, optimizer, num_epochs, model_name, num_tests):
        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=100)
            inference(model, test, num_tests, FIG_DIR, device)
            if (epoch+1)%10 == 0:
                inference(model, test, num_tests, os.path.join(FIG_DIR,model_name,f'{epoch+1}'), device)
                torch.save(model, os.path.join(LOG_DIR, f'{model_name}.pt'))

    LOG_DIR = args.log_dir
    FIG_DIR = os.path.join(LOG_DIR, 'figs')

    # train on the GPU or on the CPU, if a GPU is not available
    print(f'Cuda: {torch.cuda.is_available()}')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ### BUILD DATASET ###
    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    data_dir = args.data_dir
    train = MPFDataset(os.path.join(data_dir, 'train'), get_transform(train=True))
    test = MPFDataset(os.path.join(data_dir, 'test'), get_transform(train=False))

    # define training and validation data loaders
    train_data_loader = torch.utils.data.DataLoader(
        train,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        collate_fn=utils.collate_fn
    )

    test_data_loader = torch.utils.data.DataLoader(
        test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn
    )
    ### BUILD MODEL ###
    # get the model using our helper function
    NEW_ANC=False
    LARGE_MH=True
    model = get_model_instance_segmentation(num_classes, 0, new_anchors=NEW_ANC, large_mask_head=LARGE_MH)

    print(len([p for p in model.parameters()]))
    params = [p for p in model.parameters() if p.requires_grad]
    print(len(params))


    # move model to the right device
    model.to(device)
    inference(model, test, 1, FIG_DIR, device)
    # construct an optimizer
    optimizer = torch.optim.AdamW(
        params,
        lr=1e-3
    )
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.1
    )
    ### TRAINING ###
    print('starting training')
    try:
       train_model(model, train_data_loader, optimizer, 20, 'train_heads', 15)
    except KeyboardInterrupt as e:
        pass
    

    model2 = get_model_instance_segmentation(num_classes, 3, new_anchors=NEW_ANC, large_mask_head=LARGE_MH)
    params = [p for p in model2.parameters() if p.requires_grad]
    model2.to(device)
    model2.load_state_dict(model.state_dict())
    del model
    del train_data_loader 

    torch.cuda.empty_cache()

    train_data_loader = torch.utils.data.DataLoader(
        train,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        collate_fn=utils.collate_fn
    )

    optimizer = torch.optim.AdamW(
        params,
        lr=1e-4
    )

    try:
        train_model(model2,  train_data_loader, optimizer, 40, 'train_bb3', 15)
    except KeyboardInterrupt as e:
        pass



    model3 = get_model_instance_segmentation(num_classes, 5, new_anchors=NEW_ANC, large_mask_head=LARGE_MH)
    params = [p for p in model3.parameters() if p.requires_grad]
    model3.to(device)
    model3.load_state_dict(model2.state_dict())
    del model2
    torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(
        params,
        lr=1e-4
    )

    try:
        train_model(model3, train_data_loader, optimizer, 40, 'train_bb5', 15)
    except KeyboardInterrupt as e:
        pass
