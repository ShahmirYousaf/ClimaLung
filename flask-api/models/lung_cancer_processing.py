import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import SimpleITK as sitk
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
from models.CNNCABM import LungCancerBinaryClassifier
from models.lung_mask_generation import LungMaskGenerator
from pathlib import Path

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
    def __init__(self, ct_scan_path, mask, normalize=True, augment=False):
        print("INSIDE THIS PART inside start of lidc idri")
        self.ct_path = ct_scan_path
        self.mask_path = mask
        self.normalize = normalize
        self.augment = augment
        # self.label=label


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

       
        
        if Path(self.ct_path).suffix.lower() != '.npy':
            ct_scan=cv2.imread(self.ct_path, cv2.IMREAD_GRAYSCALE)
        else:
            ct_scan=np.load(self.ct_path)

        nodule_mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        nodule_mask = np.where(nodule_mask > 0, 1, 0).astype(np.uint8)

    # Resize everything to 512x512 upfront
        ct_scan = cv2.resize(ct_scan, (512, 512), interpolation=cv2.INTER_LINEAR)
        nodule_mask = cv2.resize(nodule_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
       
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

        # if self.label.lower()== 'true':
        #     label = 1
        # elif self.label.lower() == 'false':
        #     label = 0
       
   
        return ct_scan_tensor, nodule_mask_tensor, radiomics_features, ct_scan,nodule_mask, radiomics_features_

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
        print("INSIDE THIS PART")
        """Crop the lung region to focus on relevant parts of the CT scan"""
        mask = np.uint8(mask)
        _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            x_start, y_start, x_end, y_end = self.expand_bboxx(x, y, w, h, image.shape)
            return image[y_start:y_end, x_start:x_end], mask[y_start:y_end, x_start:x_end]  
            print("INSIDE THIS PART")
        else:
            print("no countours found")
        return image, mask

    def expand_bboxx(self, x, y, w, h, img_shape, padding=60):
        """Expand the bounding box with padding"""
        x_start = max(x - padding, 0)
        y_start = max(y - padding, 0)
        x_end = min(x + w + padding, img_shape[1])
        y_end = min(y + h + padding, img_shape[0])
        return x_start, y_start, x_end, y_end




def GetPrediction(ct_scan_path, mask_file_path):

    processor = LungMaskGenerator()
    processor.process_and_save_mask(ct_scan_path, mask_file_path)


    dataset=LIDCDataset(ct_scan_path, mask_file_path)
    ct_scan, nodule_mask, radiomics, full_scan, full_mask, rads=dataset[0]
    # ct_display = ct_scan.squeeze().numpy()  # Remove channel dim (1,224,224) -> (224,224)
    # mask_display = nodule_mask.squeeze().numpy()  # Remove channel dim if present


    # print(radiomics,"radiomics features: ")
    model=load_model()

    ct_scan= ct_scan.unsqueeze(0).float().to(DEVICE)
    radiomics = torch.from_numpy(radiomics).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        output = model(ct_scan, radiomics)
        prob = torch.sigmoid(output).item()
        print("prob of cancer", prob)
        pred = 1 if prob > 0.55 else 0

    print(f"predicition: {pred} ")


#     feature_keys = [
#     "Mean", "Energy", "Entropy", "Kurtosis", "Skewness", "Variance",
#     "Elongation", "Sphericity", "Perimeter",
#     "Contrast", "Correlation"
#      radiomics_dict = dict(zip(feature_keys, radiomics))

    scan_base64 = highlight_nodule_in_scan(full_scan, full_mask)
    print("INSIDE THIS PART")
    # mask_base64 = mark_nodule_on_full_scan(full_scan, full_mask)

    print("dictionarryyyyyy", rads)

    medical_insights = generate_medical_insights(rads, pred, prob)
    
    return {
        'scan_image': scan_base64,
        'prediction': pred,
        'medical_insights': medical_insights
    }


    img = Image.open(BytesIO(img_data))
    
    # Display the image using matplotlib
    plt.imshow(img)
    plt.axis('off')
    plt.title("Nodule Detection Result")
    plt.show()


    print(ct_display.shape)
    plt.figure(figsize=(10, 5))

    # Display CT scan
    plt.subplot(1, 2, 1)
    plt.imshow(ct_display, cmap='gray')
    plt.title(f'CT Scan (Label: {label})')
    plt.axis('off')

    # Display Nodule Mask
    plt.subplot(1, 2, 2)
    plt.imshow(mask_display, cmap='gray')
    plt.title('Nodule Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()



def generate_medical_insights(radiomics,label, prob):
    """Generate comprehensive clinical insights from radiomics features with label integration"""
    insights = []
    
    # 1. Comprehensive Morphological Analysis (now label-sensitive)
    morph_findings = []
    
    perimeter = radiomics.get('Perimeter', 0)
    size_description = (f"substantial size (Perimeter: {perimeter:.1f} mm)" if perimeter > 50 else
                       f"moderate dimensions (Perimeter: {perimeter:.1f} mm)" if perimeter > 20 else
                       f"relatively small size (Perimeter: {perimeter:.1f} mm)")
    
    if label == 1 and perimeter > 30:
        size_description += " (concerning for malignancy given size)"
    morph_findings.append(size_description)

    sphericity = radiomics.get('Sphericity', 0)
    elongation = radiomics.get('Elongation', 0)
    
    shape_analysis = ("highly spherical morphology suggesting well-circumscribed nature" 
                     if sphericity > 0.85 and elongation < 1.2 else
                     "irregular, elongated configuration with potential lobulations/spiculations" 
                     if sphericity < 0.7 and elongation > 1.5 else
                     "intermediate shape complexity")
    
    if label == 1 and (sphericity < 0.7 or elongation > 1.5):
        shape_analysis += " - malignant features present"
    morph_findings.append(shape_analysis)
    
    insights.append({
        'title': 'Detailed Morphological Profile',
        'findings': morph_findings
    })
    
    # 2. Advanced Texture Characterization (enhanced with label context)
    texture_findings = []
    
    entropy = radiomics.get('Entropy', 0)
    entropy_text = f"entropy value of {entropy:.2f} indicates {describe_entropy_pattern(entropy)}"
    if label == 1 and entropy > 5:
        entropy_text += " (typical of malignant lesions)"
    texture_findings.append(entropy_text)
    
    contrast = radiomics.get('Contrast', 0)
    correlation = radiomics.get('Correlation', 0)
    
    texture_pattern = ("marked textural heterogeneity with poor structural organization" 
                      if contrast > 15 and correlation < 0.3 else
                      "uniform texture with highly organized internal architecture" 
                      if contrast < 5 and correlation > 0.7 else
                      "moderate textural variation with some structural regularity")
    
    if label == 1 and (contrast > 10 or correlation < 0.4):
        texture_pattern += " - suspicious for malignancy"
    texture_findings.append(texture_pattern)
    
    insights.append({
        'title': 'Quantitative Texture Evaluation',
        'findings': texture_findings
    })
    
    # 3. Statistical Feature Analysis (label-adjusted)
    stat_findings = []
    
    skewness = radiomics.get('Skewness', 0)
    skewness_text = (f"asymmetric intensity distribution (Skewness: {skewness:.2f})" 
                    if abs(skewness) > 1 else 
                    f"relatively symmetric intensity distribution (Skewness: {skewness:.2f})")
    
    if label == 1 and abs(skewness) > 0.8:
        skewness_text += " (common in malignant nodules)"
    stat_findings.append(skewness_text)
    
    kurtosis = radiomics.get('Kurtosis', 0)
    kurtosis_text = ("peaked intensity distribution suggesting focal variations" 
                    if kurtosis > 3.5 else 
                    "flattened intensity distribution" 
                    if kurtosis < 2.5 else 
                    "moderate intensity distribution")
    
    if label == 1 and kurtosis > 3:
        kurtosis_text += " - malignant pattern observed"
    stat_findings.append(kurtosis_text)
    
    insights.append({
        'title': 'Statistical Distribution Patterns',
        'findings': stat_findings
    })
    
    # 4. Clinical Integration with Label-Aware Risk Stratification
    clinical_findings = []

    if label==0:
        risk_score=0
    else:
        risk_score = round(prob,2)
    
    risk_category = "high" if risk_score > 0.7 else "intermediate" if risk_score > 0.4 else "low"
    
    print(prob, "risk score")
    risk_text = f"Composite malignancy risk score: {risk_score:.2f} ({risk_category} risk category)"
    if label is not None:
        risk_text += f" [Pathology: {'Malignant' if label == 1 else 'Benign'}]"
    clinical_findings.append(risk_text)
    
    # Label-sensitive red flags
    concerning_features = []
    if (sphericity < 0.65 and entropy > 5.5):
        concerning_features.append("irregular shape with high heterogeneity")
    if (skewness < -1.2 and contrast > 12):
        concerning_features.append("asymmetric texture with high contrast")
    
    if concerning_features:
        concern_text = "Concerning features: " + ", ".join(concerning_features)
        if label == 1:
            concern_text += " (consistent with malignancy)"
        clinical_findings.append(concern_text)
    
    # Label-guided recommendations
    recommendations = []
    if risk_category == "high":
        recommendations.append("Tissue sampling strongly recommended")
        if label == 1:
            recommendations.append("Immediate diagnostic workup indicated")
    elif risk_category == "intermediate":
        recommendations.append("Short-term follow-up imaging (3-6 months) advised")
        if label == 1:
            recommendations.append("Consider accelerated follow-up schedule")
    else:
        recommendations.append("Routine follow-up may be sufficient")
        if label == 1:
            recommendations.append("Review recommended - low risk score contradicts pathology")
    
    clinical_findings.append("Management considerations: " + "; ".join(recommendations))
    
    insights.append({
        'title': 'Clinical Correlation and Management',
        'findings': clinical_findings
    })
    
    return insights

def describe_entropy_pattern(entropy):
    """Provide nuanced description of entropy patterns"""
    return ("very high heterogeneity, potentially suggesting complex internal architecture" if entropy > 7 else
            "moderate heterogeneity, indicating some textural variation" if entropy > 5 else
            "mild heterogeneity with predominantly uniform texture" if entropy > 3 else
            "very homogeneous internal structure")

def calculate_malignancy_risk_score(radiomics, label, ):
    """Calculate composite risk score with label-informed weighting"""
    # Base weights from radiomics literature
    base_weights = {
        'Sphericity': -0.3,
        'Elongation': 0.4,
        'Entropy': 0.25,
        'Contrast': 0.15,
        'Skewness': 0.1,
        'Correlation': -0.2
    }
    
    # Label-adjusted weights
    if label == 1:  # Malignant cases
        weights = {k: v * 1.3 for k, v in base_weights.items()}  # Amplify malignant features
        weights.update({'Entropy': 0.35, 'Contrast': 0.25})  # Override specific weights
    elif label == 0:  # Benign cases
        weights = {k: v * 0.7 for k, v in base_weights.items()}  # Attenuate benign features
    else:
        weights = base_weights
    
    # Feature normalization (unchanged)
    features = {
        'Sphericity': radiomics.get('Sphericity', 0.5),
        'Elongation': radiomics.get('Elongation', 1.0),
        'Entropy': radiomics.get('Entropy', 3.0),
        'Contrast': radiomics.get('Contrast', 5.0),
        'Skewness': abs(radiomics.get('Skewness', 0.0)),
        'Correlation': radiomics.get('Correlation', 0.5)
    }
    
    normalized = {
        'Sphericity': max(0, min(1, (features['Sphericity'] - 0.5) / 0.5)),
        'Elongation': max(0, min(1, (features['Elongation'] - 1.0) / 2.0)),
        'Entropy': max(0, min(1, (features['Entropy'] - 3.0) / 4.0)),
        'Contrast': max(0, min(1, (features['Contrast'] - 5.0) / 15.0)),
        'Skewness': max(0, min(1, features['Skewness'] / 1.5)),
        'Correlation': max(0, min(1, (0.8 - features['Correlation']) / 0.8))
    }
    
    # Calculate weighted score with label consideration
    score = sum(weights[feat] * normalized[feat] for feat in weights)
    
    # Label-based score adjustment
    if label == 1:
        score = min(1.0, score * 2)  # Increase risk for known malignant cases
    elif label == 0:
        score = 0  # Decrease risk for known benign cases
    
    return score
    # return max(0, min(1, (score + 1) / 2) ) # Ensure 0-1 range











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
        

        
        
       
