import cv2
import numpy as np
import matplotlib.pyplot as plt

class GearInspectionSystemV2:
    def __init__(self, ideal_image_data):
        self.ideal_image = ideal_image_data
        if self.ideal_image is None:
            raise ValueError("Could not load ideal image data")
        
        if len(self.ideal_image.shape) == 3:
            self.ideal_image = cv2.cvtColor(self.ideal_image, cv2.COLOR_BGR2GRAY)
        
        self.ideal_processed = self.preprocess_image(self.ideal_image)
        self.ideal_inner_radius = self.calculate_inner_radius(self.ideal_processed)
        self.ideal_teeth_count = self.count_teeth_advanced(self.ideal_processed)
        
        print(f"Ideal gear processed: {self.ideal_teeth_count} teeth, inner radius: {self.ideal_inner_radius:.2f} pixels")
    
    def preprocess_image(self, image):
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        return binary
    
    def calculate_inner_radius(self, binary_image):
        inverted = cv2.bitwise_not(binary_image)
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0
        
        height, width = binary_image.shape
        center = np.array([width // 2, height // 2])
        
        min_distance = float('inf')
        inner_hole_contour = None
        
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                distance = np.linalg.norm(center - np.array([cx, cy]))
                area = cv2.contourArea(contour)
                if distance < min_distance and area > 50:
                    min_distance = distance
                    inner_hole_contour = contour
        
        if inner_hole_contour is not None:
            area = cv2.contourArea(inner_hole_contour)
            radius = np.sqrt(area / np.pi)
            return radius
        
        return 0
    
    def count_teeth_advanced(self, binary_image):
        height, width = binary_image.shape
        center = (width // 2, height // 2)
        
        angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        outer_radius = min(width, height) // 2 - 10
        
        profile = []
        for angle in angles:
            x = int(center[0] + outer_radius * np.cos(angle))
            y = int(center[1] + outer_radius * np.sin(angle))
            
            if 0 <= x < width and 0 <= y < height:
                profile.append(binary_image[y, x])
            else:
                profile.append(0)
        
        profile = np.array(profile)
        
        kernel_size = 7
        kernel = np.ones(kernel_size) / kernel_size
        smooth_profile = np.convolve(profile, kernel, mode='same')
        
        threshold = np.max(smooth_profile) * 0.8
        teeth_count = 0
        in_tooth = False
        min_tooth_width = 8
        current_tooth_width = 0
        
        for intensity in smooth_profile:
            if intensity > threshold:
                if not in_tooth:
                    in_tooth = True
                    current_tooth_width = 1
                else:
                    current_tooth_width += 1
            else:
                if in_tooth and current_tooth_width >= min_tooth_width:
                    teeth_count += 1
                in_tooth = False
                current_tooth_width = 0
        
        if in_tooth and current_tooth_width >= min_tooth_width:
            teeth_count += 1
        
        return teeth_count
    
    def detect_defects(self, sample_image_data, sample_name):
        if sample_image_data is None:
            return {"error": f"Could not load sample image: {sample_name}"}
        
        if len(sample_image_data.shape) == 3:
            sample_image = cv2.cvtColor(sample_image_data, cv2.COLOR_BGR2GRAY)
        else:
            sample_image = sample_image_data.copy()
        
        sample_processed = self.preprocess_image(sample_image)
        sample_inner_radius = self.calculate_inner_radius(sample_processed)
        sample_teeth_count = self.count_teeth_advanced(sample_processed)
        
        radius_difference = sample_inner_radius - self.ideal_inner_radius
        
        if abs(radius_difference) < 5:
            diameter_status = "identical"
        elif radius_difference > 5:
            diameter_status = "larger"
        else:
            diameter_status = "smaller"
        
        missing_teeth = max(0, self.ideal_teeth_count - sample_teeth_count)
        
        if self.ideal_processed.shape != sample_processed.shape:
            sample_processed = cv2.resize(sample_processed, 
                                        (self.ideal_processed.shape[1], self.ideal_processed.shape[0]))
        
        diff = cv2.bitwise_xor(self.ideal_processed, sample_processed)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
        diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        diff_contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        worn_teeth = 0
        total_wear_area = 0
        
        height, width = sample_processed.shape
        center = np.array([width // 2, height // 2])
        
        for contour in diff_contours:
            area = cv2.contourArea(contour)
            
            if area > 100:
                contour_center = np.mean(contour.reshape(-1, 2), axis=0)
                center_distance = np.linalg.norm(contour_center - center)
                
                if center_distance > sample_inner_radius + 30:
                    total_wear_area += area
                    if area > 300:
                        worn_teeth += 1
        
        broken_teeth = missing_teeth
        
        if total_wear_area > 1000 and missing_teeth == 0:
            worn_teeth = max(1, int(total_wear_area / 500))
        
        return {
            "sample_name": sample_name,
            "broken_teeth": broken_teeth,
            "worn_teeth": worn_teeth,
            "total_defects": broken_teeth + worn_teeth,
            "diameter_status": diameter_status,
            "inner_radius_difference": radius_difference,
            "ideal_teeth_count": self.ideal_teeth_count,
            "sample_teeth_count": sample_teeth_count,
            "total_wear_area": total_wear_area
        }

def run_gear_inspection_v2():
    try:
        ideal_data = window.fs.readFile('Image 1', {'encoding': None})
        ideal_array = np.frombuffer(ideal_data, dtype=np.uint8)
        ideal_image = cv2.imdecode(ideal_array, cv2.IMREAD_COLOR)
        
        if ideal_image is None:
            print("Error: Could not decode ideal image")
            return
        
        print("🔧 GEAR INSPECTION SYSTEM V2")
        print("=" * 60)
        inspector = GearInspectionSystemV2(ideal_image)
        
        sample_descriptions = {
            'Image 2': 'Sample 2: Large inner opening',
            'Image 3': 'Sample 3: Large inner opening + Missing teeth + worn out teeth',
            'Image 4': 'Sample 4: Missing inner opening',
            'Image 5': 'Sample 5: Missing inner opening + worn out teeth',
            'Image 6': 'Sample 6: Missing teeth'
        }
        
        results = []
        
        for i in range(2, 7):
            image_name = f'Image {i}'
            try:
                sample_data = window.fs.readFile(image_name, {'encoding': None})
                sample_array = np.frombuffer(sample_data, dtype=np.uint8)
                sample_image = cv2.imdecode(sample_array, cv2.IMREAD_COLOR)
                
                if sample_image is None:
                    print(f"Error: Could not decode {image_name}")
                    continue
                
                result = inspector.detect_defects(sample_image, image_name)
                results.append(result)
                
                description = sample_descriptions.get(image_name, image_name)
                print(f"\n🔍 INSPECTING: {description}")
                print("-" * 50)
                
                if "error" in result:
                    print(f"ERROR: {result['error']}")
                else:
                    print(f"Broken teeth: {result['broken_teeth']}")
                    print(f"Worn teeth: {result['worn_teeth']}")
                    print(f"Total defects: {result['total_defects']}")
                    print(f"Inner diameter: {result['diameter_status']}")
                    
                    if result['diameter_status'] != 'identical':
                        print(f"Radius difference: {result['inner_radius_difference']:.2f} pixels")
                    
                    print(f"Teeth count - Ideal: {result['ideal_teeth_count']}, Sample: {result['sample_teeth_count']}")
                    
                    if result['total_wear_area'] > 0:
                        print(f"Wear area detected: {result['total_wear_area']:.0f} pixels²")
                
            except Exception as e:
                print(f"Error processing {image_name}: {str(e)}")
        
        print("\n" + "=" * 60)
        print("INSPECTION SUMMARY")
        print("=" * 60)
        
        if results:
            total_samples = len(results)
            total_broken = sum(r.get('broken_teeth', 0) for r in results)
            total_worn = sum(r.get('worn_teeth', 0) for r in results)
            defective_gears = sum(1 for r in results if r.get('total_defects', 0) > 0)
            
            print(f"Total samples inspected: {total_samples}")
            print(f"Gears with defects: {defective_gears}/{total_samples}")
            print(f"Total broken teeth found: {total_broken}")
            print(f"Total worn teeth found: {total_worn}")
            
            print(f"\nDETAILED RESULTS:")
            for result in results:
                if 'error' not in result:
                    status = "PASS" if result['total_defects'] == 0 else "FAIL"
                    print(f"   {result['sample_name']}: {status} | "
                          f"Broken: {result['broken_teeth']}, "
                          f"Worn: {result['worn_teeth']}, "
                          f"Diameter: {result['diameter_status']}")
        
        print("\nINSPECTION COMPLETE!")
        
        return results
        
    except Exception as e:
        print(f"System Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def visualize_inspection_results(results):
    if not results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    sample_names = [r['sample_name'] for r in results if 'error' not in r]
    broken_counts = [r['broken_teeth'] for r in results if 'error' not in r]
    worn_counts = [r['worn_teeth'] for r in results if 'error' not in r]
    diameter_status = [r['diameter_status'] for r in results if 'error' not in r]
    
    axes[0, 0].bar(sample_names, broken_counts, color='red', alpha=0.7)
    axes[0, 0].set_title('Broken Teeth Count')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    axes[0, 1].bar(sample_names, worn_counts, color='orange', alpha=0.7)
    axes[0, 1].set_title('Worn Teeth Count')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    diameter_counts = pd.Series(diameter_status).value_counts()
    axes[1, 0].pie(diameter_counts.values, labels=diameter_counts.index, autopct='%1.1f%%')
    axes[1, 0].set_title('Inner Diameter Status Distribution')
    
    total_defects = [r['total_defects'] for r in results if 'error' not in r]
    axes[1, 1].bar(sample_names, total_defects, color='purple', alpha=0.7)
    axes[1, 1].set_title('Total Defects per Sample')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def main():
    results = run_gear_inspection_v2()
    if results:
        visualize_inspection_results(results)

if __name__ == "__main__":
    main()

run_gear_inspection_v2()