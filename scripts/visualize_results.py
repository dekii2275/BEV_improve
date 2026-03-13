import os
import cv2
import glob

def parse_kitti_label(label_path):
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split(' ')
        if len(parts) >= 15 and parts[0] != 'DontCare':
            name = parts[0]
            # KITTI format: [left, top, right, bottom]
            bbox = [float(x) for x in parts[4:8]]
            bboxes.append({'name': name, 'bbox': bbox})
    return bboxes

def draw_bboxes(img, bboxes, color, title):
    # Make a copy of the image to draw on
    img_copy = img.copy()
    for obj in bboxes:
        name = obj['name']
        bbox = obj['bbox']
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        # Draw bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        # Draw class name
        cv2.putText(img_copy, name, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    
    # Put image title
    cv2.putText(img_copy, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    return img_copy

def main():
    # Defaul paths based on standalone evaluation
    image_dir = "data/dair-v2x-i-kitti/training/image_2"
    gt_dir = "data/dair-v2x-i-kitti/training/label_2"
    pred_dir = "outputs/data"
    output_dir = "outputs/visualizations"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all images
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return
        
    count = 0
    max_images = 10 # Number of images to visualize
    
    print(f"Generating visualizations in {output_dir}...")
    for img_path in image_paths:
        if count >= max_images:
            break
            
        file_name = os.path.basename(img_path)
        label_name = file_name.replace('.jpg', '.txt')
        
        gt_path = os.path.join(gt_dir, label_name)
        pred_path = os.path.join(pred_dir, label_name)
        
        # Check if GT and Pred files exist
        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            continue
            
        # Read Original Image
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Parse labels
        gt_bboxes = parse_kitti_label(gt_path)
        pred_bboxes = parse_kitti_label(pred_path)
        
        # Draw Original
        img_orig = img.copy()
        cv2.putText(img_orig, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw Prediction (Red Bounding Box)
        img_pred = draw_bboxes(img, pred_bboxes, (0, 0, 255), "Prediction Model")
        
        # Draw Ground Truth (Green Bounding Box)
        img_gt = draw_bboxes(img, gt_bboxes, (0, 255, 0), "Ground Truth (Dataset)")
        
        # Concatenate images horizontally (side-by-side)
        combined = cv2.hconcat([img_orig, img_pred, img_gt])
        
        # Save output image
        out_path = os.path.join(output_dir, file_name)
        cv2.imwrite(out_path, combined)
        print(f"[{count+1}/{max_images}] Saved: {out_path}")
        
        count += 1
        
    print(f"Done! Evaluated and saved {count} visual comparisons.")

if __name__ == "__main__":
    main()
