import matplotlib.pyplot as plt
import numpy as np
import config
import torch
import cv2
import os

def prepare_plot(orig_image, orig_mask, pred_mask):
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    
    ax[0].imshow(orig_image)
    ax[0].set_title("Image")
    ax[1].imshow(orig_mask)
    ax[0].set_title("Original Mask")
    ax[2].imshow(pred_mask)
    ax[0].set_title("Predicted Mask")
    
    figure.tight_layout()
    figure.show()
    
def make_predictions(model, image_path):
    model.eval()
    
    with torch.no_grad():
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0
        
        image = cv2.resize(image, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT))
        orig = image.copy()
        
        filename = image_path.split(os.path.sep)[-1]
        gt_path = os.path.join(config.MASKS_DATASET_PATH, filename)
        
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_HEIGHT))
        
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(config.DEVICE)
        
        pred_mask = model(image).squeeze()
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = pred_mask.cpu().numpy()
        
        pred_mask = (pred_mask > config.THRESHOLD) * 255
        pred_mask = pred_mask.astype(np.uint8)
        
        prepare_plot(orig, gt_mask, pred_mask)
        
print("[INFO] Loading up Test Image Paths")
image_paths = open(config.TEST_PATHS).read().strip().split("\n")
image_paths = np.random.choice(image_paths, size=10)

print("[INFO] Load Model...")
model = torch.load(config.MODEL_PATH).to(config.DEVICE)

for path in image_paths:
    make_predictions(model=model, image_path=path)
