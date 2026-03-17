import os
import shutil
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as T_F
from tqdm import tqdm

from models.hed_network import load_pretrained_hed

def precompute_hed_edges(data_root, split, img_height, img_width, recompute=False, device='cuda'):
    input_dir = os.path.join(data_root, split, "images")
    output_dir = os.path.join(data_root, split, f"edges_{img_height}x{img_width}") 
    
    if recompute and os.path.exists(output_dir):
        print(f"[Config] Clearing {output_dir}...")
        shutil.rmtree(output_dir)
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
        
    filenames = sorted(os.listdir(input_dir))
    if not recompute and len(os.listdir(output_dir)) == len(filenames):
        return output_dir
        
    print(f"⚙️ Computing HED Edges ({img_height}x{img_width}) for {split}...")
    hed_model = load_pretrained_hed().to(device).eval()
    hed_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        for fname in tqdm(filenames, desc=f"HED {split}"):
            save_path = os.path.join(output_dir, os.path.splitext(fname)[0] + ".png")
            if os.path.exists(save_path) and not recompute: continue
            try:
                img = Image.open(os.path.join(input_dir, fname)).convert("RGB")
                inp = hed_transform(img).unsqueeze(0).to(device)
                edge_map = hed_model(inp)
                edge_np = (edge_map.squeeze().cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(edge_np).save(save_path)
            except Exception as e:
                print(f"Error {fname}: {e}")
                
    return output_dir

class EdgeAwareDataset(Dataset):
    def __init__(self, root_dir, split="train", img_height=630, img_width=350, mask_extension=".png", edge_dir=None):
        self.split_dir = os.path.join(root_dir, split)
        self.image_dir = os.path.join(self.split_dir, "images")
        self.mask_dir = os.path.join(self.split_dir, "masks")
        self.edge_dir = edge_dir
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.split = split
        self.mask_extension = mask_extension
        self.img_height = img_height
        self.img_width = img_width

        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.base_transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        self.edge_transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor()
        ])
        self.mask_resize = transforms.Resize((img_height, img_width), interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self): 
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        base_name = os.path.splitext(img_name)[0]
        
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, base_name + self.mask_extension)).convert("L")
        edge = Image.open(os.path.join(self.edge_dir, base_name + ".png")).convert("L")

        if self.split == "train":
            if random.random() < 0.5:
                image = T_F.hflip(image); mask = T_F.hflip(mask); edge = T_F.hflip(edge)
            if random.random() < 0.5:
                image = T_F.vflip(image); mask = T_F.vflip(mask); edge = T_F.vflip(edge)
            if random.random() < 0.75:
                angle = random.uniform(-10, 10)
                image = T_F.rotate(image, angle, transforms.InterpolationMode.BILINEAR)
                mask = T_F.rotate(mask, angle, transforms.InterpolationMode.NEAREST)
                edge = T_F.rotate(edge, angle, transforms.InterpolationMode.BILINEAR)

        img_tensor = self.base_transform(image)
        mask_tensor = torch.from_numpy((np.array(self.mask_resize(mask)) > 0).astype(np.int64))
        edge_tensor = self.edge_transform(edge)

        # Return name as well for testing loop compatibility
        if self.split == "test":
            return img_tensor, mask_tensor, edge_tensor, img_name
            
        return img_tensor, mask_tensor, edge_tensor