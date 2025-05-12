import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import SimpleITK as sitk
import radiomics 
from radiomics import firstorder, glcm, glrlm, glszm, shape2D
import cv2
import matplotlib.pyplot as plt
import logging
import warnings
import shutil
import requests
import tqdm
import tempfile
import atexit
from io import BytesIO
import base64
from PIL import Image

logging.getLogger('radiomics').setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=UserWarning, message="GLCM is symmetrical")

# MODEL_URL = "https://drive.google.com/uc?export=download&id=1OP7i8dQgWwJy_THUEZDcGQcBl8vHadCh"
# MODEL_PATH = "LungCancerModel.pth"  
# UPLOAD_FOLDER = 'LungCancerModel'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
lung_cancer_model_path = os.path.join(current_dir, '..', 'LungCancerModel')
lung_cancer_model_path = os.path.abspath(lung_cancer_model_path)

class LIDCDataset(Dataset):
    def __init__(self, ct_scan_path, mask, label, normalize=True, augment=False):
        
        self.ct_path = ct_scan_path
        self.mask_path = mask
        self.normalize = normalize
        self.augment = augment
        self.label=label


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

       
        
        if self.ct_path.lower().endswith(('.png','jpg','.jpeg')):
            ct_scan=cv2.imread(self.ct_path, cv2.IMREAD_GRAYSCALE)
        else:
            ct_scan=np.load(self.ct_path)

        nodule_mask=cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        nodule_mask = np.where(nodule_mask > 0, 1, 0).astype(np.uint8)
       
        # full_scan=ct_scan
        # nodule_mask = np.load(mask_path)
        
        # full_nodule_mask=nodule_mask# Ensure proper type
    
        # Compute radiomics features from the original CT scan and mask
        radiomics_features = self.extract_radiomics(ct_scan, nodule_mask)
        radiomics_features_=radiomics_features
        radiomics_features = np.array(list(radiomics_features.values()))
        ct_scan_crop, nodule_mask_crop = self.crop_lung_region(ct_scan, nodule_mask)
        ct_scan_crop=self.NormalizeScan(ct_scan_crop)
        ct_scan=self.NormalizeScan(ct_scan)

        # Apply base transformations (resizing and normalization)
        transformed = self.base_transform(ct_scan_crop, nodule_mask_crop)
        ct_scan_tensor = transformed['image']
        nodule_mask_tensor = transformed['mask'].unsqueeze(0)

        if self.label.lower()== 'true':
            label = 1
        elif self.label.lower() == 'false':
            label = 0
   
        return ct_scan_tensor, nodule_mask_tensor, radiomics_features, label, ct_scan,nodule_mask, radiomics_features_

    def NormalizeScan(self, image, crop_padding=20):
        
    
        MIN_BOUND = -1000.0
        MAX_BOUND = 400.0
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image>1] = 1.
        image[image<0] = 0.
        return image


    def base_transform(self, ct_scan, nodule_mask):
        """Base transformation: resizing and normalization"""
        transform = albu.Compose([
            albu.Resize(224, 224),  # Resize to 224x224 for both image and mask
            # albu.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()  # Convert to tensor during the base transformation
        ])
        # Return the entire transformed dictionary
        return transform(image=ct_scan, mask=nodule_mask)

    def extract_radiomics(self, ct_image_path, nodule_mask_path):
        """Extract radiomics features from the original CT image and mask"""
        ct_image = sitk.GetImageFromArray(ct_image_path)  # Use SimpleITK for reading CT scan
        nodule_mask = sitk.GetImageFromArray(nodule_mask_path)

        extracted_features = {}

        # Extract first-order features
        firstOrderFeatures = firstorder.RadiomicsFirstOrder(ct_image, nodule_mask)
        firstorder_ = firstOrderFeatures.execute()
        extracted_features["Mean"] = firstorder_['Mean']
        extracted_features["Energy"] = firstorder_['Energy']
        extracted_features["Entropy"] = firstorder_['Entropy']
        extracted_features["Kurtosis"] = firstorder_['Kurtosis']
        extracted_features["Skewness"] = firstorder_['Skewness']
        extracted_features["Variance"] = firstorder_['Variance']

        # Extract shape features
        shapeFeatures = shape2D.RadiomicsShape2D(ct_image, nodule_mask)
        shape_ = shapeFeatures.execute()
        extracted_features["Elongation"] = shape_['Elongation']
        extracted_features["Sphericity"] = shape_['Sphericity']
        extracted_features["Perimeter"] = shape_['Perimeter']

        # Extract GLCM features
        glcmFeatures = glcm.RadiomicsGLCM(ct_image, nodule_mask)
        glcm_ = glcmFeatures.execute()
        extracted_features["Contrast"] = glcm_['Contrast']
        extracted_features["Correlation"] = glcm_['Correlation']

        return extracted_features
    def crop_lung_region(self, image, mask):
        """Crop the lung region to focus on relevant parts of the CT scan"""
        mask = np.uint8(mask)
        _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            x_start, y_start, x_end, y_end = self.expand_bboxx(x, y, w, h, image.shape)
            return image[y_start:y_end, x_start:x_end], mask[y_start:y_end, x_start:x_end]
        return image, mask

    def expand_bboxx(self, x, y, w, h, img_shape, padding=60):
        """Expand the bounding box with padding"""
        x_start = max(x - padding, 0)
        y_start = max(y - padding, 0)
        x_end = min(x + w + padding, img_shape[1])
        y_end = min(y + h + padding, img_shape[0])
        return x_start, y_start, x_end, y_end



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return x * out.view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.conv(concat)
        return x * self.sigmoid(spatial_out)

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
    
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class LungCancerBinaryClassifier(nn.Module):
    def __init__(self, input_channels=1, radiomics_dim=12):
        super().__init__()
        
        # CNN Backbone with more conservative architecture
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Dropout2d(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout2d(0.3)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout2d(0.4)
        )
        
        self.pool = nn.MaxPool2d(2, 2)
        self.cbam = CBAM(128)
        
        # Flattened size calculation
        self.conv_out_features = self._get_conv_output_features(input_channels)
        
        # Radiomics Pathway with more conservative architecture
        self.radiomics_fc = nn.Sequential(
            nn.Linear(radiomics_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.4))
        
        # Classifier for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(self.conv_out_features + 64, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # Single output for binary classification
        )
        
        self._initialize_weights()

    def _get_conv_output_features(self, input_channels):
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 224, 224)
            dummy_output = self._forward_features(dummy_input)
            return dummy_output.view(1, -1).size(1)

    def _forward_features(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.cbam(x)
        return x

    def _initialize_weights(self):
        
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Using relu as proxy
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x, radiomics_features):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        radiomics = self.radiomics_fc(radiomics_features.float())
        combined = torch.cat((x, radiomics), dim=1)
        return torch.sigmoid(self.classifier(combined))  # Sigmoid for binary classification



    

def GetPrediction(ct_scan_path, mask,label_):

    dataset=LIDCDataset(ct_scan_path, mask,label_)
    ct_scan, nodule_mask, radiomics,label, full_scan, full_mask, rads=dataset[0]
    # ct_display = ct_scan.squeeze().numpy()  # Remove channel dim (1,224,224) -> (224,224)
    # mask_display = nodule_mask.squeeze().numpy()  # Remove channel dim if present


    # print(radiomics,"radiomics features: ")
    model=load_model()

    ct_scan= ct_scan.unsqueeze(0).float().to(DEVICE)
    radiomics = torch.from_numpy(radiomics).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        output = model(ct_scan, radiomics)
        prob = torch.sigmoid(output).item()
        pred = 1 if prob > 0.5 else 0

    print(f"predicition: {pred} label : {label}")


#     feature_keys = [
#     "Mean", "Energy", "Entropy", "Kurtosis", "Skewness", "Variance",
#     "Elongation", "Sphericity", "Perimeter",
#     "Contrast", "Correlation"
# ]

#     radiomics_dict = dict(zip(feature_keys, radiomics))

    scan_base64 = highlight_nodule_in_scan(full_scan, full_mask)
    # mask_base64 = mark_nodule_on_full_scan(full_scan, full_mask)

    print("dictionarryyyyyy", rads)

    medical_insights = generate_medical_insights(rads, label)
    
    return {
        'scan_image': scan_base64,
        'prediction': label,
        'medical_insights': medical_insights
    }


    # img = Image.open(BytesIO(img_data))
    
    # # Display the image using matplotlib
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("Nodule Detection Result")
    # plt.show()


    # print(ct_display.shape)
    # plt.figure(figsize=(10, 5))

    # # Display CT scan
    # plt.subplot(1, 2, 1)
    # plt.imshow(ct_display, cmap='gray')
    # plt.title(f'CT Scan (Label: {label})')
    # plt.axis('off')

    # # Display Nodule Mask
    # plt.subplot(1, 2, 2)
    # plt.imshow(mask_display, cmap='gray')
    # plt.title('Nodule Mask')
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()

def generate_medical_insights(radiomics, label):
    """Generate comprehensive clinical insights from radiomics features"""
    insights = []
    
    # 1. Comprehensive Morphological Analysis
    morph_findings = []
    
    # Perimeter-based size assessment
    perimeter = radiomics.get('Perimeter', 0)
    
    if perimeter > 50:
        morph_findings.append(f"substantial size (Perimeter: {perimeter:.1f} mm)")
    elif perimeter > 20:
        morph_findings.append(f"moderate dimensions (Perimeter: {perimeter:.1f} mm)")
    else:
        morph_findings.append(f"relatively small size (Perimeter: {perimeter:.1f} mm)")
    
    # Shape characterization
    sphericity = radiomics.get('Sphericity', 0)
    elongation = radiomics.get('Elongation', 0)
    
    if sphericity > 0.85 and elongation < 1.2:
        morph_findings.append("highly spherical morphology suggesting well-circumscribed nature")
    elif sphericity < 0.7 and elongation > 1.5:
        morph_findings.append("irregular, elongated configuration with potential lobulations/spiculations")
    else:
        morph_findings.append("intermediate shape complexity")
    
    insights.append({
        'title': 'Detailed Morphological Profile',
        'findings': morph_findings
    })
    
    # 2. Advanced Texture Characterization
    texture_findings = []
    
    entropy = radiomics.get('Entropy', 0)
    texture_findings.append(
        f"entropy value of {entropy:.2f} indicates {describe_entropy_pattern(entropy)}"
    )
    
    contrast = radiomics.get('Contrast', 0)
    correlation = radiomics.get('Correlation', 0)
    
    if contrast > 15 and correlation < 0.3:
        texture_findings.append("marked textural heterogeneity with poor structural organization")
    elif contrast < 5 and correlation > 0.7:
        texture_findings.append("uniform texture with highly organized internal architecture")
    else:
        texture_findings.append("moderate textural variation with some structural regularity")
    
    insights.append({
        'title': 'Quantitative Texture Evaluation',
        'findings': texture_findings
    })
    
    # 3. Statistical Feature Analysis
    stat_findings = []
    
    skewness = radiomics.get('Skewness', 0)
    kurtosis = radiomics.get('Kurtosis', 0)
    
    if abs(skewness) > 1:
        stat_findings.append(f"asymmetric intensity distribution (Skewness: {skewness:.2f})")
    else:
        stat_findings.append(f"relatively symmetric intensity distribution (Skewness: {skewness:.2f})")
    
    if kurtosis > 3.5:
        stat_findings.append("peaked intensity distribution suggesting focal variations")
    elif kurtosis < 2.5:
        stat_findings.append("flattened intensity distribution")
    
    insights.append({
        'title': 'Statistical Distribution Patterns',
        'findings': stat_findings
    })
    
    # 4. Clinical Integration with Risk Stratification
    clinical_findings = []
    
    risk_score = calculate_malignancy_risk_score(radiomics)
    risk_category = "high" if risk_score > 0.7 else "intermediate" if risk_score > 0.4 else "low"
    
    clinical_findings.append(
        f"Composite malignancy risk score: {risk_score:.2f} ({risk_category} risk category)"
    )
    
    # Feature-specific red flags
    if (sphericity < 0.65 and entropy > 5.5) or (skewness < -1.2 and contrast > 12):
        clinical_findings.append("Multiple concerning features present including irregular shape and textural heterogeneity")
    
    recommendations = []
    if risk_category == "high":
        recommendations.append("Tissue sampling strongly recommended")
        recommendations.append("Consider PET-CT for metabolic evaluation if clinically indicated")
    elif risk_category == "intermediate":
        recommendations.append("Short-term follow-up imaging (3-6 months) advised")
        recommendations.append("Multidisciplinary review recommended")
    else:
        recommendations.append("Routine follow-up may be sufficient")
    
    clinical_findings.append("Management considerations: " + "; ".join(recommendations))
    
    insights.append({
        'title': 'Clinical Correlation and Management',
        'findings': clinical_findings
    })
    
    return insights

def describe_entropy_pattern(entropy):
    """Provide nuanced description of entropy patterns"""
    if entropy > 7:
        return "very high heterogeneity, potentially suggesting complex internal architecture"
    elif entropy > 5:
        return "moderate heterogeneity, indicating some textural variation"
    elif entropy > 3:
        return "mild heterogeneity with predominantly uniform texture"
    else:
        return "very homogeneous internal structure"

def calculate_malignancy_risk_score(radiomics):
    """Calculate composite risk score based on multiple features"""
    weights = {
        'Sphericity': -0.3,
        'Elongation': 0.4,
        'Entropy': 0.25,
        'Contrast': 0.15,
        'Skewness': 0.1,
        'Correlation': -0.2
    }
    
    # Get feature values with defaults
    features = {
        'Sphericity': radiomics.get('Sphericity', 0.5),
        'Elongation': radiomics.get('Elongation', 1.0),
        'Entropy': radiomics.get('Entropy', 3.0),
        'Contrast': radiomics.get('Contrast', 5.0),
        'Skewness': abs(radiomics.get('Skewness', 0.0)),
        'Correlation': radiomics.get('Correlation', 0.5)
    }
    
    # Calculate normalized values (0-1 range)
    normalized = {
        'Sphericity': (features['Sphericity'] - 0.5) / 0.5,  # 0-1 range for 0.5-1.0 sphericity
        'Elongation': (features['Elongation'] - 1.0) / 2.0,  # 0-1 range for 1.0-3.0 elongation
        'Entropy': (features['Entropy'] - 3.0) / 4.0,       # 0-1 range for 3.0-7.0 entropy
        'Contrast': (features['Contrast'] - 5.0) / 15.0,    # 0-1 range for 5.0-20.0 contrast
        'Skewness': min(features['Skewness'], 1.5) / 1.5,   # 0-1 range for 0-1.5 skewness
        'Correlation': (0.8 - features['Correlation']) / 0.8  # 0-1 range for 0-0.8 correlation
    }
    
    # Clip all values to 0-1 range
    for key in normalized:
        normalized[key] = max(0, min(1, normalized[key]))
    
    # Calculate weighted score
    score = sum(weights[feat] * normalized[feat] for feat in weights)
    
    # Scale to 0-1 range
    return max(0, min(1, (score + 1) / 2))

# Helper functions for medical descriptions
def describe_size(perimeter):
    size = perimeter / np.pi  # Approximate diameter
    if size < 5: return "a small focus (<5mm)"
    elif 5 <= size < 10: return "a subcentimeter dimension (5-9mm)"
    elif 10 <= size < 20: return "an intermediate size (10-19mm)"
    else: return "a large configuration (≥20mm)"

def describe_shape(sphericity, elongation):
    if sphericity > 0.9: return "a highly spherical morphology"
    elif elongation > 1.5: return "an elongated, irregular contour"
    else: return "moderately complex borders"

def describe_glcm(correlation):
    if correlation > 0.8: return "high structural uniformity"
    elif correlation > 0.5: return "moderate tissue regularity"
    else: return "disorganized tissue patterns"




def convert_to_base64(scan):
    """ Convert the CT scan to a base64 encoded PNG """
    # Normalize the scan to the range [0, 255]
    min_val = np.min(scan)
    max_val = np.max(scan)

    # Avoid division by zero
    if max_val - min_val == 0:
        scan = np.zeros_like(scan)
    else:
        scan = (scan - min_val) / (max_val - min_val) * 255.0

    # Convert to uint8
    scan = scan.astype(np.uint8)

    # Convert to RGB (grayscale 3-channel for consistency)
    rgb_image = np.stack([scan] * 3, axis=-1)

    # Save as PNG and convert to base64
    buffer = BytesIO()
    image = Image.fromarray(rgb_image)
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode('utf-8')


def highlight_nodule_in_scan(full_scan, full_mask, overlay_color=(255, 0, 0), overlay_alpha=0.3):
    """
    Highlights nodule regions in the original CT scan with a semi-transparent overlay.
    
    Args:
        full_scan: 2D numpy array of the CT scan (any dtype)
        full_mask: 2D numpy array of the nodule mask (binary or 0-1 values)
        overlay_color: RGB color for the highlight (default: red)
        overlay_alpha: Opacity of the highlight (0-1)
    
    Returns:
        base64 encoded PNG image string
    """
    # Normalize scan to 0-255 uint8 if needed
    if full_scan.dtype != np.uint8:
        scan_min = full_scan.min()
        scan_max = full_scan.max()
        if scan_max - scan_min > 0:
            full_scan = (255 * (full_scan - scan_min) / (scan_max - scan_min)).astype(np.uint8)
        else:
            full_scan = np.zeros_like(full_scan, dtype=np.uint8)
    
    # Ensure mask is binary (0 and 1)
    full_mask = (full_mask > 0).astype(np.uint8)
    
    # Create RGB image from the grayscale scan
    rgb_image = np.stack([full_scan] * 3, axis=-1)
    
    # Create overlay with the specified color
    overlay = np.zeros_like(rgb_image)
    overlay[..., 0] = overlay_color[0]  # R
    overlay[..., 1] = overlay_color[1]  # G
    overlay[..., 2] = overlay_color[2]  # B
    
    # Blend the original image with overlay where mask is 1
    highlighted = rgb_image.copy()
    mask_indices = full_mask > 0
    highlighted[mask_indices] = (overlay_alpha * overlay[mask_indices] + 
                                (1 - overlay_alpha) * rgb_image[mask_indices])
    
    # Convert to PIL Image and then to base64
    image = Image.fromarray(highlighted)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return img_base64





    
def load_model():
    try:
        model_dir=lung_cancer_model_path+'\model2.pth'
        model=LungCancerBinaryClassifier(input_channels=1, radiomics_dim=11)
        model.to(DEVICE)
        checkpoint = torch.load(model_dir, map_location=DEVICE, weights_only=False)


        if 'model_state_dict' not in checkpoint:
            print("Checkpoint missing model state dict")
            return None
            
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # Try alternative loading methods
        try:
            import pickle
            with open(MODEL_PATH, 'rb') as f:
                checkpoint = pickle.load(f)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print("Model loaded via alternative method!")
            return model
        except:
            print("All loading methods failed")
            return None
        

        
        
       
