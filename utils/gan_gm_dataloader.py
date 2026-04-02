import torch
from torch.utils.data import Dataset
import os
from torchvision.transforms import transforms
from PIL import Image
import random
# No need for nibabel or numpy if only handling standard 2D images

from torchvision.transforms import transforms as T # Use alias to avoid conflict

# Helper to load standard 2D image as PIL (L)
def load_image_pil(path):
    """Helper function to load standard 2D image files (PNG, JPG, etc.) into PIL Image (L)."""
    try:
        # Open and convert to grayscale (L)
        img = Image.open(path).convert('L')
        return img
    except Exception as e:
        # Provide more context if loading fails
        print(f"Error loading image file {path}: {e}")
        raise # Re-raise the exception


class MRIDataset(Dataset):
    """
    用于加载 AD/NC 图像对及其对应的灰质图的数据集类 (全部为 2D 图片)。
    CycleGAN 通常使用非配对数据，但这里我们加载配对的原图和灰质图。
    """
    def __init__(self, root_ad, root_nc, root_ad_gm, root_nc_gm, transform, mode='train', unaligned=False):
        self.transform = transform
        self.transform_gm = transform
        self.unaligned = unaligned
        self.mode = mode

        # Define allowed file extensions for standard 2D images
        self.allowed_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

        try:
            # Load original image files
            self.files_ad = sorted([os.path.join(root_ad, f) for f in os.listdir(root_ad) if f.lower().endswith(self.allowed_extensions)])
            self.files_nc = sorted([os.path.join(root_nc, f) for f in os.listdir(root_nc) if f.lower().endswith(self.allowed_extensions)])

            # Load gray matter map files
            self.files_ad_gm = sorted([os.path.join(root_ad_gm, f) for f in os.listdir(root_ad_gm) if f.lower().endswith(self.allowed_extensions)])
            self.files_nc_gm = sorted([os.path.join(root_nc_gm, f) for f in os.listdir(root_nc_gm) if f.lower().endswith(self.allowed_extensions)])

        except FileNotFoundError as e:
            # Make sure to print the directory name that caused the error
            error_dir = e.filename if hasattr(e, 'filename') else "unknown directory"
            raise FileNotFoundError(f"Dataset directory not found: {e.strerror} in {error_dir}")
        except Exception as e:
            raise RuntimeError(f"Error listing files in dataset directories: {e}")

        # Validate file counts and presence
        if not self.files_ad:
            raise ValueError(f"No image files found in AD directory: {root_ad}. Allowed extensions: {', '.join(self.allowed_extensions)}")
        if not self.files_nc:
            raise ValueError(f"No image files found in NC directory: {root_nc}. Allowed extensions: {', '.join(self.allowed_extensions)}")
        if not self.files_ad_gm:
            raise ValueError(f"No GM files found in AD GM directory: {root_ad_gm}. Allowed extensions: {', '.join(self.allowed_extensions)}")
        if not self.files_nc_gm:
            raise ValueError(f"No GM files found in NC GM directory: {root_nc_gm}. Allowed extensions: {', '.join(self.allowed_extensions)}")

        # Important: Check if original and GM files match in count (implies correspondence by sorting)
        if len(self.files_ad) != len(self.files_ad_gm):
            raise ValueError(f"Mismatch in AD file counts: {len(self.files_ad)} images vs {len(self.files_ad_gm)} GM images.")
        if len(self.files_nc) != len(self.files_nc_gm):
            raise ValueError(f"Mismatch in NC file counts: {len(self.files_nc)} images vs {len(self.files_nc_gm)} GM images.")


        print(f"Initialized {mode} dataset: {len(self.files_ad)} AD pairs, {len(self.files_nc)} NC pairs (all 2D images).")

    def __getitem__(self, index):
        try:
            # Get paths for AD pair (original + GM)
            img_ad_path = self.files_ad[index % len(self.files_ad)]
            gm_ad_path = self.files_ad_gm[index % len(self.files_ad_gm)] # Use same index as sorted lists match

            # Get paths for NC pair (original + GM)
            if self.unaligned:
                # For unaligned, pick a random NC pair
                random_nc_index = random.randint(0, len(self.files_nc) - 1)
                img_nc_path = self.files_nc[random_nc_index]
                gm_nc_path = self.files_nc_gm[random_nc_index] # GM corresponds to the chosen random NC image
            else:
                # For aligned, use the same index for NC pair
                img_nc_path = self.files_nc[index % len(self.files_nc)]
                gm_nc_path = self.files_nc_gm[index % len(self.files_nc_gm)] # Use same index

            # Load images and GM maps as PIL using the simplified loader
            img_ad_pil = load_image_pil(img_ad_path)
            gm_ad_pil = load_image_pil(gm_ad_path)
            img_nc_pil = load_image_pil(img_nc_path)
            gm_nc_pil = load_image_pil(gm_nc_path)

            # Apply transforms
            img_ad_tensor = self.transform(img_ad_pil)
            img_nc_tensor = self.transform(img_nc_pil)

            # gm_transform applies Resize and ToTensor, mapping uint8 [0, 255] to float [0, 1]
            gm_ad_tensor = self.transform_gm(gm_ad_pil)
            gm_nc_tensor = self.transform_gm(gm_nc_pil)

            return {"AD": img_ad_tensor, "NC": img_nc_tensor, "AD_GM": gm_ad_tensor, "NC_GM": gm_nc_tensor}

        except Exception as e:
            print(f"Error processing index {index}: {e}. Skipping this sample.")
            # Optionally print paths for failed sample
            # print(f"Failed paths: AD_img={img_ad_path}, AD_gm={gm_ad_path}, NC_img={img_nc_path}, NC_gm={gm_nc_path}")
            return None # Let DataLoader's collate_fn handle None

    def __len__(self):
        # Length is determined by the number of pairs in each domain
        return max(len(self.files_ad), len(self.files_nc))


# Custom collate_fn to handle None values returned by __getitem__
# This function remains the same as before
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        # print("Warning: Batch is empty after filtering bad samples.")
        return None # Return None if the batch is empty after filtering

    # Use default collate on the filtered batch
    # This correctly stacks tensors and handles dictionaries
    return torch.utils.data.dataloader.default_collate(batch)

