from sklearn.model_selection import train_test_split
from core.dataset import SegmentationDataset
from core.model import UNet
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torchvision import transforms as T
from imutils import paths
import matplotlib.pyplot as plt
import config
import torch
import time
import tqdm

def train():
    image_paths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
    mask_paths = sorted(list(paths.list_images(config.MASKS_DATASET_PATH)))
    
    split = train_test_split(image_paths, mask_paths, test_size=config.TEST_SPLIT, random_state=42)
    
    train_images, test_images = split[:2]
    train_masks, test_masks = split[2:]
    
    print("[INFO] Saving Testing Images Path...")
    f = open(config.TEST_PATHS, "w")
    f.write("\n".join(test_images))
    f.close()
    
    transforms = T.Compose(
        [
         T.ToPILImage(),
         T.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)),
         T.ToTensor()
        ]
    )
    
    train_ds = SegmentationDataset(image_paths=train_images, mask_paths=train_masks, transforms=transforms)
    test_ds = SegmentationDataset(image_paths=test_images, mask_paths=test_masks, transforms=transforms)
    print(f"[INFO] Found {len(train_ds)} examples in the training set...")
    print(f"[INFO] Found {len(test_ds)} examples in the test set...")
    
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY, num_workers=config.NUM_WORKERS)
    
    model = UNet(out_size=(config.INPUT_IMAGE_HEIGHT,  config.INPUT_IMAGE_WIDTH)).to(config.DEVICE)
    loss_fn = BCEWithLogitsLoss()
    opt = Adam(model.parameters(), lr=config.LR)
    
    train_steps = len(train_ds) // config.BATCH_SIZE
    test_steps = len(test_ds) // config.BATCH_SIZE
    
    H = {
         "train_loss": [],
         "test_loss": []
    }
    
    print("[INFO] Training the Network...")
    start_time = time.time()
    
    for e in tqdm.tqdm(range(config.NUM_EPOCHS)):
        model.train()
        
        total_train_loss = 0
        total_test_loss = 0
        
        for index, (x, y) in enumerate(train_loader):
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)
            
            pred = model(x)
            loss = loss_fn(pred, y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_train_loss += loss
            
        with torch.no_grad():
            model.eval()
            
            for x, y in test_loader:
                x = x.to(config.DEVICE)
                y = y.to(config.DEVICE)
                
                pred = model(x)
                total_test_loss += loss_fn(pred, y)
                
        avg_train_loss = total_train_loss / train_steps
        avg_test_loss = total_test_loss / test_steps
        
        H["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        H["test_loss"].append(avg_test_loss.cpu().detach().numpy())
        
        print("[INFO] Epoch: {}/{}".format(e+1, config.NUM_EPOCHS))
        print("[INFO] Train Loss: {:.6f}, Test Loss: {:.6f}".format(avg_train_loss, avg_test_loss))
        
    end_time = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(end_time- start_time))
    
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="Train Loss")
    plt.plot(H["test_loss"], label="Test Loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(config.PLOT_PATH)
    
    print("[INFO] Saving the Model...")
    torch.save(model, config.MODEL_PATH)
    
# Required to stop the RuntimeError  freeze_support() from happening
if __name__ == "__main__":
    train()
