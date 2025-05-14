from models.Unet import UNet
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torchvision import transforms
import io
from typing import Tuple, Union
import argparse
import os
import numpy as np
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torchvision import transforms
from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import median_filter
from skimage import measure, morphology
import scipy.ndimage as ndimage
from sklearn.cluster import KMeans

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
lung_segmentation_model = os.path.join(current_dir, '..', 'LungCancerModel')
lung_segmentation_model_path = os.path.abspath(lung_segmentation_model)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def segment_lung(img):
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    
    middle = img[100:400,100:400] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
   
    img[img==max]=mean
    img[img==min]=mean
    
   
    img= median_filter(img,size=3)
   
    img= anisotropic_diffusion(img)
    
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
    eroded = morphology.erosion(thresh_img,np.ones([4,4]))
    dilation = morphology.dilation(eroded,np.ones([10,10]))
    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
            good_labels.append(prop.label)
    mask = np.ndarray([512,512],dtype=np.int8)
    mask[:] = 0
   
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) 
    return mask*img





class LungMaskGenerator:
    def __init__(self):
        print("inside")
        model_dir=lung_segmentation_model_path +'\checkpoint.pth'
        self.model = UNet(n_channels=1, n_classes=1, bilinear=True)

        checkpoint=torch.load(model_dir,map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from various formats (npy, jpg, jpeg, jfif)"""
        if image_path.endswith('.npy'):
            ct_scan = np.load(image_path)
        else:
      
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            ct_scan = np.array(img)

            ct_scan = (ct_scan / 255.0) * 1400 - 1000
            
        return ct_scan
    
    def normalize_scan(self, image: np.ndarray) -> np.ndarray:
        """Normalize CT scan to 0-1 range"""
        MIN_BOUND = -1000.0
        MAX_BOUND = 400.0
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image > 1] = 1.
        image[image < 0] = 0.
        return image
    
    def preprocess(self, ct_scan: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess the CT scan including lung segmentation"""
        # Ensure the image is 512x512 (resize if needed)
        if ct_scan.shape != (512, 512):
            ct_scan = cv2.resize(ct_scan, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        # Store full scan before lung segmentation
        full_scan = self.normalize_scan(ct_scan.copy())
        
        # Apply lung segmentation
        segmented_scan = segment_lung(ct_scan)
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        segmented_scan = transform(segmented_scan).float()
        full_scan = transform(full_scan).float()
        
        return segmented_scan.unsqueeze(0).to(DEVICE), full_scan.unsqueeze(0)
    
    def predict(self, image_path: str, threshold: float = 0.5) -> dict:
        
       
        ct_scan = self.load_image(image_path)
        input_tensor, full_scan_tensor = self.preprocess(ct_scan)

       
        with torch.no_grad():
            output = self.model(input_tensor)
            output_mask = torch.sigmoid(output)
            binary_mask = (output_mask > threshold).float()
        
        return {
            'input_scan': input_tensor.squeeze().cpu().numpy(),
            'full_scan': full_scan_tensor.squeeze().cpu().numpy(),
            'predicted_mask': binary_mask.squeeze().cpu().numpy(),
            'predicted_probabilities': output_mask.squeeze().cpu().numpy()
        }
    
    def mask_to_png(self, mask: np.ndarray) -> bytes:
        """Convert numpy mask to PNG bytes"""
        # Scale mask to 0-255 range
        mask_image = (mask * 255).astype(np.uint8)
        
        # Create PIL Image
        pil_img = Image.fromarray(mask_image)
        
        # Save to bytes buffer
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='PNG')
        
        return img_byte_arr.getvalue()
    
    def save_mask_as_png(self, mask: np.ndarray, output_path: str) -> None:
        """Save the mask as a PNG file"""
        png_data = self.mask_to_png(mask)
        with open(output_path, 'wb') as f:
            f.write(png_data)
    
    def process_and_save_mask(self, image_path: str, output_path: str, threshold: float = 0.5) -> dict:
        """Process image and save the predicted mask as PNG"""
        results = self.predict(image_path, threshold)
        self.save_mask_as_png(results['predicted_mask'], output_path)
        return results
    
    def visualize(self, image_path: str, threshold: float = 0.5, save_mask: bool = False, mask_output_path: str = None) -> dict:
        """Visualize the original scan, lung segmentation, and predicted nodules"""
        results = self.predict(image_path, threshold)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Full CT scan
        axes[0].imshow(results['full_scan'], cmap='gray')
        axes[0].set_title("Original CT Scan")
        axes[0].axis('off')
        
        # Lung segmented scan (model input)
        axes[1].imshow(results['input_scan'], cmap='gray')
        axes[1].set_title("Lung Segmented Scan")
        axes[1].axis('off')
        
        # Predicted nodule mask
        axes[2].imshow(results['predicted_mask'], cmap='gray')
        axes[2].set_title("Predicted Nodule Mask")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        if save_mask:
            if mask_output_path is None:
                mask_output_path = image_path.rsplit('.', 1)[0] + '_mask.png'
            self.save_mask_as_png(results['predicted_mask'], mask_output_path)
        
        return results