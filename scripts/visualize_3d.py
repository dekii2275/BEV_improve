import os
import cv2
import glob
import math
import numpy as np

def get_lidar_3d_8points(obj_size, yaw_lidar, center_lidar):
    center_lidar = [float(center_lidar[0]), float(center_lidar[1]), float(center_lidar[2])]
    lidar_r = np.matrix([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], 
                         [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], 
                         [0, 0, 1]])
    l, w, h = obj_size
    center_lidar[2] = center_lidar[2] - h / 2
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = lidar_r * corners_3d_lidar + np.matrix(center_lidar).T
    return corners_3d_lidar.T

def box3d_to_image_points(box3d, Tr_velo_to_cam, P2):
    corners_3d = np.array(box3d)
    corners_3d_extend = np.concatenate(
        [corners_3d, np.ones((corners_3d.shape[0], 1), dtype=np.float32)], axis=1) 
    corners_3d_extend = np.matmul(Tr_velo_to_cam, corners_3d_extend.transpose(1, 0))
    corners_2d = np.matmul(P2, corners_3d_extend)
    pts_2d = corners_2d[:2] / corners_2d[2]
    valid_points = corners_2d[2] > 0
    return pts_2d, valid_points

def draw_projected_box3d(image, pts_2d, valid_points, color, thickness=2):
    if valid_points.sum() < 8:
         return
    pts = pts_2d.T.astype(int)
    
    # Draw bottom face
    cv2.line(image, tuple(pts[0]), tuple(pts[1]), color, thickness)
    cv2.line(image, tuple(pts[1]), tuple(pts[2]), color, thickness)
    cv2.line(image, tuple(pts[2]), tuple(pts[3]), color, thickness)
    cv2.line(image, tuple(pts[3]), tuple(pts[0]), color, thickness)
    
    # Draw top face
    cv2.line(image, tuple(pts[4]), tuple(pts[5]), color, thickness)
    cv2.line(image, tuple(pts[5]), tuple(pts[6]), color, thickness)
    cv2.line(image, tuple(pts[6]), tuple(pts[7]), color, thickness)
    cv2.line(image, tuple(pts[7]), tuple(pts[4]), color, thickness)
    
    # Draw vertical lines connecting top and bottom faces
    cv2.line(image, tuple(pts[0]), tuple(pts[4]), color, thickness)
    cv2.line(image, tuple(pts[1]), tuple(pts[5]), color, thickness)
    cv2.line(image, tuple(pts[2]), tuple(pts[6]), color, thickness)
    cv2.line(image, tuple(pts[3]), tuple(pts[7]), color, thickness)

    # Cross to indicate the front:
    # Based on get_lidar_3d_8points: 0,1,4,5 are X > 0 (front of car in lidar coordinates)
    cv2.line(image, tuple(pts[0]), tuple(pts[5]), color, thickness)
    cv2.line(image, tuple(pts[1]), tuple(pts[4]), color, thickness)

def parse_kitti_calib_full(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    ret = {}
    for line in lines:
        if line.startswith('P2:'):
            ret['P2'] = np.array([float(x) for x in line.split(' ')[1:]]).reshape(3, 4)
        elif line.startswith('Tr_velo_to_cam:'):
            Tr_velo_to_cam = np.zeros((4, 4))
            Tr_velo_to_cam[:3, :4] = np.array(line.split(' ')[1:]).astype(float).reshape(3,4)
            Tr_velo_to_cam[3, 3] = 1
            ret['Tr_velo_to_cam'] = Tr_velo_to_cam
    return ret

def draw_bboxes(img, label_path, calib_P2, Tr_cam2lidar, Tr_velo_to_cam, color, title):
    img_copy = img.copy()
    if not os.path.exists(label_path):
        cv2.putText(img_copy, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        return img_copy
        
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split(' ')
        if len(parts) < 15 or parts[0] == 'DontCare':
            continue
            
        name = parts[0]
        # In DAIR-v2x eval, indices are: 8=dh, 9=dw, 10=dl
        dh = float(parts[8])
        dw = float(parts[9])
        dl = float(parts[10])
        lx, ly, lz = float(parts[11]), float(parts[12]), float(parts[13])
        ry = float(parts[14])
        
        # Follow result2kitti.py exact logic:
        yaw_lidar = 0.5 * np.pi - ry
        loc_cam = np.array([lx, ly, lz, 1.0]).reshape(4, 1)
        loc_lidar = np.matmul(Tr_cam2lidar, loc_cam).squeeze(-1)[:3]
        loc_lidar[2] += 0.5 * dh
        
        center_lidar = loc_lidar
        obj_size = [dl, dw, dh]
        
        box3d = get_lidar_3d_8points(obj_size, yaw_lidar, center_lidar)
        
        pts_2d, valid_points = box3d_to_image_points(box3d, Tr_velo_to_cam, calib_P2)
        valid_arr = np.asarray(valid_points).flatten()
        
        draw_projected_box3d(img_copy, pts_2d, valid_arr, color, 2)
        
        pts_2d_arr = np.asarray(pts_2d)
        if valid_arr.all():
            text_x, text_y = int(pts_2d_arr[0, 4]), int(pts_2d_arr[1, 4]) # Point 4 x, y
            cv2.putText(img_copy, name, (text_x, max(text_y - 5, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                        
    cv2.putText(img_copy, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    return img_copy

def main():
    image_dir = "data/dair-v2x-i-kitti/training/image_2"
    gt_dir = "data/dair-v2x-i-kitti/training/label_2"
    calib_dir = "data/dair-v2x-i-kitti/training/calib"
    pred_dir = "outputs/data"
    output_dir = "outputs/visualizations_3d"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # We should iterate over predictions available (validation set)
    pred_paths = sorted(glob.glob(os.path.join(pred_dir, '*.txt')))
    if not pred_paths:
        print(f"No predictions found in {pred_dir}")
        return
        
    count = 0
    max_images = 10 
    
    print(f"Generating 3D visualizations in {output_dir}...")
    for pred_path in pred_paths:
        if count >= max_images:
            break
            
        file_name = os.path.basename(pred_path)
        base_name = file_name.replace('.txt', '')
        
        img_path = os.path.join(image_dir, f"{base_name}.jpg")
        gt_path = os.path.join(gt_dir, f"{base_name}.txt")
        calib_path = os.path.join(calib_dir, f"{base_name}.txt")
        
        img = cv2.imread(img_path)
        if img is None: continue
            
        calib_data = parse_kitti_calib_full(calib_path)
        if 'P2' not in calib_data or 'Tr_velo_to_cam' not in calib_data:
            continue
            
        calib_P2 = calib_data['P2']
        Tr_velo_to_cam = calib_data['Tr_velo_to_cam']
        Tr_cam2lidar = np.linalg.inv(Tr_velo_to_cam)
        
        img_orig = img.copy()
        cv2.putText(img_orig, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        img_pred = draw_bboxes(img.copy(), pred_path, calib_P2, Tr_cam2lidar, Tr_velo_to_cam, (0, 0, 255), "Prediction 3D")
        img_gt = draw_bboxes(img.copy(), gt_path, calib_P2, Tr_cam2lidar, Tr_velo_to_cam, (0, 255, 0), "Ground GT 3D")
        
        combined = cv2.hconcat([img_orig, img_pred, img_gt])
        out_path = os.path.join(output_dir, f"{base_name}.jpg")
        cv2.imwrite(out_path, combined)
        print(f"[{count+1}/{max_images}] Saved: {out_path}")
        count += 1
        
    print(f"Done! Saved {count} 3D visual comparisons.")

if __name__ == "__main__":
    main()
