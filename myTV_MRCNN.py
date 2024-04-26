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


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    anchor_sizes = ((8,), (16,), (32,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(progress=True, 
                                                                  weights_backbone = "ResNet50_Weights.DEFAULT",
                                                                  trainable_backbone_layers=0,
                                                                  num_classes=2,
                                                                rpn_batch_size_per_image= 64,
                                                                rpn_post_nms_top_n_train= 1000,
                                                                rpn_post_nms_top_n_test= 2000,
                                                                rpn_nms_thresh= 0.9,
                                                                box_batch_size_per_image=128,
                                                                min_size=512,
                                                                max_size=512,
                                                                box_detections_per_img= 100,
                                                                )

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 1024
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        2
    ) 
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
    LOG_DIR = os.path.join('logs')
    FIG_DIR = os.path.join(LOG_DIR, 'figs')

    ### SHOW DATA EXAMPLE ###
    image = cv2.imread("Processed Data/train/IMGs/000142.png")
    mask = read_image("Processed Data/train/masks/000142.png")
    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.title("Image")
    plt.imshow(image)
    plt.subplot(122)
    plt.title("Mask")
    plt.imshow(mask.permute(1, 2, 0))
    plt.savefig(fname = os.path.join(FIG_DIR, 'Im and Mask.png'))
    plt.close()

    # train on the GPU or on the CPU, if a GPU is not available
    print(f'Cuda: {torch.cuda.is_available()}')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ### BUILD DATASET ###
    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    train = MPFDataset('Processed Data/train', get_transform(train=True))
    test = MPFDataset('Processed Data/test', get_transform(train=False))

    # define training and validation data loaders
    train_data_loader = torch.utils.data.DataLoader(
        train,
        batch_size=8,
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
    model = get_model_instance_segmentation(num_classes)
    print(len([p for p in model.parameters()]))
    for p in model.parameters():
        p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    print(len(params))

    # move model to the right device
    model.to(device)

    # construct an optimizer
    optimizer = torch.optim.SGD(
        params,
        lr=3e-4,
        momentum=0.9,
        weight_decay=0.0005
    )
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.1
    )
    ### TRAINING ###
    num_epochs = 8
    print('starting training')
    try:
        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10)
            if (epoch+1)%4 == 0:
                torch.save(model, os.path.join(LOG_DIR, f'model{epoch+1}.pt'))
            # update the learning rate
            # evaluate on the test dataset
            # evaluate(model, data_loader_test, device=device)
    except KeyboardInterrupt as e:
        pass

    ### EVALUATION ON 1 IMAGE ###
    print('starting evaluation')
    image, targ = test.__getitem__(np.random.randint(0,50))

    eval_transform = get_transform(train=False)
    print(torch.min(image))

    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        plt.imshow(x.squeeze(0))
        plt.show()
        x = x.to(device)
        
        predictions = model([x])
        pred = predictions[0]


    # image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    # image = image.reshape(512,608,3)
    # image = image[:3, ...]
    print(image.shape)
    # pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    # pred_boxes = pred["boxes"].long()
    # output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    # output_image = draw_segmentation_masks(image, masks, alpha=0.5, colors="blue")

    masks = (pred['masks'] > 0.6)
    print(f'{len(pred["boxes"])=}')
    plt.close()
    fig, ax = plt.subplots(ncols = 3)
    fig.set_figwidth(12)
    fig.set_figheight(6)
    ax[0].imshow(image.squeeze(0))
    ax[0].set_title('source')
    ax[1].set_title('source targets')
    ax[1].imshow(torch.sum(targ['masks'], dim=0))
    for rect in targ['boxes']:
            x1, y1, x2, y2 = rect
            width = x2 - x1
            height = y2 - y1
            rect_patch = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax[1].add_patch(rect_patch)
    im = ax[2].imshow(torch.Tensor.cpu(torch.sum(masks, dim = 0).squeeze(0)))
    ax[2].set_title('predicted')
    for i in range(len(pred['boxes'])):
            if pred['scores'][i] > 0.65:
                rect = pred['boxes'][i]
                x1, y1, x2, y2 = torch.Tensor.cpu(rect)
                width = x2 - x1
                height = y2 - y1
                rect_patch = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
                ax[2].add_patch(rect_patch)
    plt.savefig(fname = os.path.join(FIG_DIR, 'pred_check.png'))
