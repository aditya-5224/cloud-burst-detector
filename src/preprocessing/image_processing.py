"""
Image Processing Pipeline

Handles cloud detection, binary mask generation, and feature extraction
from satellite imagery for cloud burst prediction.
"""

import cv2
import numpy as np
import pandas as pd
import yaml
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
from scipy import ndimage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """Main class for satellite image processing and feature extraction"""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        """Initialize the image processor"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_config = self.config['features']['image_features']
        self.processed_data_path = Path(self.config['data']['processed_data_path'])
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
    def generate_cloud_mask(self, 
                           cloud_probability: np.ndarray,
                           threshold: float = 50.0) -> np.ndarray:
        """
        Generate binary cloud mask from cloud probability data
        
        Args:
            cloud_probability: Array with cloud probability values (0-100)
            threshold: Probability threshold for cloud detection
            
        Returns:
            Binary mask where 1 = cloud, 0 = no cloud
        """
        
        # Create binary mask
        cloud_mask = (cloud_probability >= threshold).astype(np.uint8)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Remove noise with opening
        cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_OPEN, kernel)
        
        # Fill holes with closing
        cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_CLOSE, kernel)
        
        logger.info(f"Generated cloud mask with {np.sum(cloud_mask)} cloud pixels")
        return cloud_mask
    
    def calculate_cloud_coverage(self, cloud_mask: np.ndarray) -> float:
        """
        Calculate cloud coverage percentage
        
        Args:
            cloud_mask: Binary cloud mask
            
        Returns:
            Cloud coverage percentage (0-100)
        """
        
        total_pixels = cloud_mask.size
        cloud_pixels = np.sum(cloud_mask)
        coverage = (cloud_pixels / total_pixels) * 100
        
        return coverage
    
    def extract_glcm_features(self, 
                             image: np.ndarray,
                             distances: List[int] = [1, 2],
                             angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4]) -> Dict[str, float]:
        """
        Extract Gray-Level Co-occurrence Matrix (GLCM) texture features
        
        Args:
            image: Grayscale image array
            distances: List of pixel distances
            angles: List of angles in radians
            
        Returns:
            Dictionary with GLCM texture features
        """
        
        # Ensure image is uint8 and in correct range
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # Calculate GLCM
        glcm = graycomatrix(image, distances=distances, angles=angles, 
                           levels=256, symmetric=True, normed=True)
        
        # Extract properties
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        features = {
            'glcm_contrast': float(contrast),
            'glcm_dissimilarity': float(dissimilarity),
            'glcm_homogeneity': float(homogeneity),
            'glcm_energy': float(energy),
            'glcm_correlation': float(correlation)
        }
        
        return features
    
    def count_cloud_blobs(self, cloud_mask: np.ndarray, min_size: int = 50) -> Dict[str, int]:
        """
        Count and analyze cloud blobs/regions
        
        Args:
            cloud_mask: Binary cloud mask
            min_size: Minimum size for a blob to be counted
            
        Returns:
            Dictionary with blob statistics
        """
        
        # Label connected components
        labeled_mask = label(cloud_mask)
        regions = regionprops(labeled_mask)
        
        # Filter regions by size
        large_regions = [r for r in regions if r.area >= min_size]
        
        blob_stats = {
            'blob_count': len(large_regions),
            'total_blob_area': sum(r.area for r in large_regions),
            'largest_blob_area': max([r.area for r in large_regions], default=0),
            'average_blob_area': np.mean([r.area for r in large_regions]) if large_regions else 0
        }
        
        return blob_stats
    
    def calculate_infrared_statistics(self, image: np.ndarray) -> Dict[str, float]:
        """
        Calculate infrared band statistics (simulated for cloud probability data)
        
        Args:
            image: Image array
            
        Returns:
            Dictionary with infrared statistics
        """
        
        # For cloud probability data, we'll calculate temperature-like statistics
        # In real implementation, this would use actual infrared bands
        
        stats = {
            'infrared_mean': float(np.mean(image)),
            'infrared_std': float(np.std(image)),
            'infrared_min': float(np.min(image)),
            'infrared_max': float(np.max(image)),
            'infrared_median': float(np.median(image)),
            'infrared_percentile_25': float(np.percentile(image, 25)),
            'infrared_percentile_75': float(np.percentile(image, 75))
        }
        
        return stats
    
    def extract_edge_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract edge-based features using various edge detection methods
        
        Args:
            image: Grayscale image array
            
        Returns:
            Dictionary with edge features
        """
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # Canny edge detection
        edges_canny = cv2.Canny(image, 50, 150)
        
        # Sobel edge detection
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        edges_sobel = np.sqrt(grad_x**2 + grad_y**2)
        
        # Laplacian edge detection
        edges_laplacian = cv2.Laplacian(image, cv2.CV_64F)
        
        features = {
            'canny_edge_density': float(np.sum(edges_canny > 0) / edges_canny.size),
            'sobel_edge_mean': float(np.mean(edges_sobel)),
            'sobel_edge_std': float(np.std(edges_sobel)),
            'laplacian_variance': float(np.var(edges_laplacian)),
            'gradient_magnitude_mean': float(np.mean(np.sqrt(grad_x**2 + grad_y**2)))
        }
        
        return features
    
    def extract_all_features(self, 
                           cloud_probability: np.ndarray,
                           cloud_threshold: float = 50.0) -> Dict[str, float]:
        """
        Extract all image-based features from cloud probability data
        
        Args:
            cloud_probability: Array with cloud probability values
            cloud_threshold: Threshold for cloud mask generation
            
        Returns:
            Dictionary with all extracted features
        """
        
        features = {}
        
        # Generate cloud mask
        cloud_mask = self.generate_cloud_mask(cloud_probability, cloud_threshold)
        
        # Cloud coverage
        features['cloud_coverage_percentage'] = self.calculate_cloud_coverage(cloud_mask)
        
        # GLCM texture features
        glcm_features = self.extract_glcm_features(cloud_probability)
        features.update(glcm_features)
        
        # Blob analysis
        blob_features = self.count_cloud_blobs(cloud_mask)
        features.update(blob_features)
        
        # Infrared statistics
        ir_features = self.calculate_infrared_statistics(cloud_probability)
        features.update(ir_features)
        
        # Edge features
        edge_features = self.extract_edge_features(cloud_probability)
        features.update(edge_features)
        
        # Additional statistical features
        features.update({
            'cloud_probability_mean': float(np.mean(cloud_probability)),
            'cloud_probability_std': float(np.std(cloud_probability)),
            'cloud_probability_skewness': float(self._calculate_skewness(cloud_probability)),
            'cloud_probability_kurtosis': float(self._calculate_kurtosis(cloud_probability)),
            'cloud_compactness': self._calculate_cloud_compactness(cloud_mask),
            'cloud_uniformity': self._calculate_cloud_uniformity(cloud_probability)
        })
        
        logger.info(f"Extracted {len(features)} image features")
        return features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_cloud_compactness(self, cloud_mask: np.ndarray) -> float:
        """Calculate cloud shape compactness"""
        if np.sum(cloud_mask) == 0:
            return 0
        
        # Calculate perimeter and area
        contours, _ = cv2.findContours(cloud_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0
        
        total_area = np.sum(cloud_mask)
        total_perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)
        
        if total_perimeter == 0:
            return 0
        
        # Compactness = 4π * area / perimeter²
        compactness = (4 * np.pi * total_area) / (total_perimeter ** 2)
        return float(compactness)
    
    def _calculate_cloud_uniformity(self, cloud_probability: np.ndarray) -> float:
        """Calculate cloud probability uniformity"""
        hist, _ = np.histogram(cloud_probability, bins=50, range=(0, 100))
        hist = hist / np.sum(hist)  # Normalize
        
        # Calculate uniformity as inverse of entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        uniformity = 1 / (1 + entropy)
        
        return float(uniformity)
    
    def process_image_batch(self, 
                           images: List[np.ndarray],
                           region_name: str = 'default') -> pd.DataFrame:
        """
        Process a batch of images and extract features
        
        Args:
            images: List of image arrays
            region_name: Name of the region
            
        Returns:
            DataFrame with extracted features for all images
        """
        
        features_list = []
        
        for i, image in enumerate(images):
            try:
                features = self.extract_all_features(image)
                features['image_id'] = i
                features['region'] = region_name
                features['timestamp'] = pd.Timestamp.now()
                features_list.append(features)
                
                logger.info(f"Processed image {i+1}/{len(images)}")
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                continue
        
        # Create DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Save to file
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_features_{region_name}_{timestamp}.csv"
        filepath = self.processed_data_path / filename
        features_df.to_csv(filepath, index=False)
        
        logger.info(f"Saved features for {len(features_df)} images to {filepath}")
        return features_df


def main():
    """Main function for running image processing"""
    processor = ImageProcessor()
    
    # Generate mock image data for testing
    np.random.seed(42)
    mock_images = []
    
    for i in range(5):
        # Generate mock cloud probability data
        image = np.random.beta(2, 5, (100, 100)) * 100
        mock_images.append(image)
    
    # Process the batch
    features_df = processor.process_image_batch(mock_images, 'test_region')
    print(f"Extracted features shape: {features_df.shape}")
    print(f"Feature columns: {list(features_df.columns)}")


if __name__ == "__main__":
    main()