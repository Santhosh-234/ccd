"""
Enhanced CarDD-based damage detection
Provides realistic damage detection without requiring the full CarDD dataset
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import random

class CarDDDetector:
    """Enhanced damage detection with CarDD-inspired algorithms"""
    
    def __init__(self):
        self.damage_types = {
            'dent': 0,
            'scratch': 0,
            'crack': 0,
            'paint_damage': 0,
            'bumper_damage': 0,
            'glass_damage': 0
        }
        
    def detect_damage(self, image_bytes: bytes) -> Optional[Dict]:
        """Detect damage using enhanced computer vision techniques"""
        try:
            # Convert bytes to image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return None
            
            # Enhanced damage detection
            damage_regions = []
            damage_types = self.damage_types.copy()
            
            # Analyze image for different damage types
            regions = self._analyze_damage_patterns(image)
            
            for region in regions:
                damage_type = region['type']
                confidence = region['confidence']
                
                if confidence > 0.80:  # Only high-confidence detections
                    damage_types[damage_type] += 1
                    damage_regions.append({
                        'class': damage_type,
                        'confidence': confidence,
                        'bbox': region['bbox']
                    })
            
            # Calculate severity
            severity = self._calculate_enhanced_severity(damage_types, damage_regions)
            
            return {
                'severity': severity,
                'damage_types': damage_types,
                'regions': damage_regions,
                'total_damage': sum(damage_types.values()),
                'avg_confidence': np.mean([r['confidence'] for r in damage_regions]) if damage_regions else 0.0,
                'num_detections': len(damage_regions)
            }
            
        except Exception as e:
            print(f"ERROR: Enhanced damage detection failed: {e}")
            return None
    
    def _analyze_damage_patterns(self, image: np.ndarray) -> List[Dict]:
        """Analyze image for various damage patterns"""
        regions = []
        h, w = image.shape[:2]
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # 1. Detect dents (circular depressions)
        dents = self._detect_dents(gray, h, w)
        regions.extend(dents)
        
        # 2. Detect scratches (linear features)
        scratches = self._detect_scratches(gray, h, w)
        regions.extend(scratches)
        
        # 3. Detect cracks (high edge density)
        cracks = self._detect_cracks(gray, h, w)
        regions.extend(cracks)
        
        # 4. Detect paint damage (color variations)
        paint_damage = self._detect_paint_damage(hsv, lab, h, w)
        regions.extend(paint_damage)
        
        # 5. Detect bumper damage (front/rear impact)
        bumper_damage = self._detect_bumper_damage(image, h, w)
        regions.extend(bumper_damage)
        
        # 6. Detect glass damage (reflection patterns)
        glass_damage = self._detect_glass_damage(image, h, w)
        regions.extend(glass_damage)
        
        return regions
    
    def _detect_dents(self, gray: np.ndarray, h: int, w: int) -> List[Dict]:
        """Detect dents using circular Hough transform"""
        regions = []
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles (potential dents)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=10, maxRadius=100
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                if 20 < x < w-20 and 20 < y < h-20:  # Within image bounds
                    confidence = min(0.9, r / 50.0)  # Larger circles = higher confidence
                    regions.append({
                        'type': 'dent',
                        'confidence': confidence,
                        'bbox': [x-r, y-r, x+r, y+r]
                    })
        
        return regions
    
    def _detect_scratches(self, gray: np.ndarray, h: int, w: int) -> List[Dict]:
        """Detect scratches using line detection"""
        regions = []
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if length > 30:  # Only long lines
                    confidence = min(0.8, length / 100.0)
                    regions.append({
                        'type': 'scratch',
                        'confidence': confidence,
                        'bbox': [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                    })
        
        return regions
    
    def _detect_cracks(self, gray: np.ndarray, h: int, w: int) -> List[Dict]:
        """Detect cracks using high edge density"""
        regions = []
        
        # Enhanced edge detection
        edges = cv2.Canny(gray, 30, 100)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area for cracks
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                
                # Calculate edge density
                roi_edges = edges[y:y+h_rect, x:x+w_rect]
                edge_density = np.sum(roi_edges > 0) / (w_rect * h_rect)
                
                if edge_density > 0.1:  # High edge density indicates cracks
                    confidence = min(0.8, edge_density * 5)
                    regions.append({
                        'type': 'crack',
                        'confidence': confidence,
                        'bbox': [x, y, x+w_rect, y+h_rect]
                    })
        
        return regions
    
    def _detect_paint_damage(self, hsv: np.ndarray, lab: np.ndarray, h: int, w: int) -> List[Dict]:
        """Detect paint damage using color analysis"""
        regions = []
        
        # Analyze color variations
        hsv_std = np.std(hsv, axis=(0, 1))
        lab_std = np.std(lab, axis=(0, 1))
        
        # High color variation indicates paint damage
        color_variation = np.mean(hsv_std) + np.mean(lab_std)
        
        if color_variation > 30:  # Threshold for paint damage
            # Find regions with high color variation
            mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 255))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 200:
                    x, y, w_rect, h_rect = cv2.boundingRect(contour)
                    confidence = min(0.8, color_variation / 50.0)
                    regions.append({
                        'type': 'paint_damage',
                        'confidence': confidence,
                        'bbox': [x, y, x+w_rect, y+h_rect]
                    })
        
        return regions
    
    def _detect_bumper_damage(self, image: np.ndarray, h: int, w: int) -> List[Dict]:
        """Detect bumper damage in front/rear areas"""
        regions = []
        
        # Focus on front and rear areas (top and bottom 20% of image)
        front_region = image[:h//5, :]
        rear_region = image[4*h//5:, :]
        
        for region, name in [(front_region, 'front'), (rear_region, 'rear')]:
            if region.size > 0:
                gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray_region, 50, 150)
                
                # High edge density in bumper areas
                edge_density = np.sum(edges > 0) / edges.size
                
                if edge_density > 0.05:  # Threshold for bumper damage
                    confidence = min(0.7, edge_density * 10)
                    if name == 'front':
                        bbox = [0, 0, w, h//5]
                    else:
                        bbox = [0, 4*h//5, w, h]
                    
                    regions.append({
                        'type': 'bumper_damage',
                        'confidence': confidence,
                        'bbox': bbox
                    })
        
        return regions
    
    def _detect_glass_damage(self, image: np.ndarray, h: int, w: int) -> List[Dict]:
        """Detect glass damage using reflection analysis"""
        regions = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Look for irregular reflection patterns
        # Glass damage often shows as irregular bright spots
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 1000:  # Glass damage size range
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                
                # Check if it's in a likely glass area (windshield/windows)
                if (y < h//3 or y > 2*h//3) and (x > w//4 and x < 3*w//4):
                    confidence = min(0.6, area / 500.0)
                    regions.append({
                        'type': 'glass_damage',
                        'confidence': confidence,
                        'bbox': [x, y, x+w_rect, y+h_rect]
                    })
        
        return regions
    
    def _calculate_enhanced_severity(self, damage_types: Dict, damage_regions: List[Dict]) -> str:
        """Calculate severity using enhanced CarDD-inspired logic"""
        
        total_damage = sum(damage_types.values())
        if total_damage == 0:
            return "minor"
        
        # Weight different damage types
        severity_score = 0
        
        # Critical damage (highest weight)
        critical = damage_types['crack'] + damage_types['glass_damage']
        severity_score += critical * 4
        
        # Major damage
        major = damage_types['dent'] + damage_types['bumper_damage']
        severity_score += major * 3
        
        # Minor damage
        minor = damage_types['scratch'] + damage_types['paint_damage']
        severity_score += minor * 1
        
        # Factor in confidence
        if damage_regions:
            avg_confidence = np.mean([r['confidence'] for r in damage_regions])
            confidence_factor = avg_confidence * 2
            severity_score += confidence_factor
        
        # Enhanced thresholds
        if severity_score >= 10:
            return "severe"
        elif severity_score >= 5:
            return "moderate"
        else:
            return "minor"
    
    def annotate_image(self, image_bytes: bytes, detection_result: Dict) -> np.ndarray:
        """Annotate image with enhanced damage detection results"""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return np.zeros((100, 100, 3), dtype=np.uint8)
            
            annotated = image.copy()
            
            # Color mapping for different damage types
            colors = {
                'dent': (0, 0, 255),           # Red
                'scratch': (0, 255, 0),        # Green
                'crack': (255, 0, 0),          # Blue
                'paint_damage': (0, 255, 255), # Yellow
                'bumper_damage': (255, 0, 255), # Magenta
                'glass_damage': (255, 255, 0)  # Cyan
            }
            
            # Draw bounding boxes and labels
            for region in detection_result['regions']:
                x1, y1, x2, y2 = region['bbox']
                class_name = region['class']
                confidence = region['confidence']
                color = colors.get(class_name, (128, 128, 128))
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Background for label
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                
                # Text
                cv2.putText(annotated, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add summary text
            summary = f"Enhanced: {detection_result['severity']} | {detection_result['total_damage']} regions"
            cv2.rectangle(annotated, (0, 0), (len(summary) * 12, 40), (0, 0, 0), -1)
            cv2.putText(annotated, summary, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return annotated
            
        except Exception as e:
            print(f"ERROR: Error annotating image: {e}")
            return image if 'image' in locals() else np.zeros((100, 100, 3), dtype=np.uint8)

def create_enhanced_detector() -> CarDDDetector:
    """Factory function to create enhanced detector"""
    return CarDDDetector()
