import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.io import read_image
import cv2
import torch 
import os
import utils
from myTV_MRCNN import get_model_instance_segmentation, get_transform, MPFDataset
from engine import evaluate

def inference(model, num_test_imgs, fig_dir, device):
    print('starting evaluation')
    os.makedirs(fig_dir, exist_ok=True)
    model.eval()
    eval_transform = get_transform(train=False)
    img_count = min(num_test_imgs,len(test))
    for idx in range(img_count):
        print(f'Index: {idx+1} / {img_count}', end='\r')
        image, targ = test.__getitem__(idx)

        with torch.no_grad():
            x = eval_transform(image)
            # convert RGBA -> RGB and move to device
            plt.imshow(x.squeeze(0))
            plt.show()
            x = x.to(device)
            
            predictions = model([x])
            pred = predictions[0]

        masks = (pred['masks'] > 0.6)
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
        plt.savefig(fname = os.path.join(fig_dir, f'{idx}.png'))

if __name__ == '__main__':
    gpu = torch.cuda.is_available()
    print(f'Cuda: {gpu}')
    device = torch.device('cuda') if gpu else torch.device('cpu')
    device = torch.device('cpu')

    NUM_CLASSES = 2
    NUM_TEST_IMGS = 10

    dir_path = os.path.dirname(__file__)
    model_path = os.path.join(dir_path, 'logs/model4.pt')
    FIG_DIR = os.path.join(dir_path, f'inference/model4')
    # os.makedirs(FIG_DIR, exist_ok=False)
    print(f'loading model...\n {model_path}')
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # dataset = MPFDataset('Processed Data/train', get_transform(train=True))
    model = get_model_instance_segmentation(NUM_CLASSES, 0)

    model.load_state_dict(torch.load(model_path, map_location=device).state_dict())

    test = MPFDataset('Processed Data/test', get_transform(train=False))
    # evaluate(model, test_data_loader, device=device)
    inference(model, NUM_TEST_IMGS, FIG_DIR, device)