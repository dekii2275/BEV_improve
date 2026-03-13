#####################
# Based on https://github.com/hongzhenwang/RRPN-revise
# Licensed under The MIT License
# Author: yanyan, scrin@foxmail.com
# Modified: CPU-only version to avoid numba.cuda segfault
#####################
import math
import numba
import numpy as np


@numba.jit(nopython=True)
def div_up(m, n):
    return m // n + (m % n > 0)


@numba.jit(nopython=True)
def _rbbox_to_corners_cpu(corners, rbbox):
    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]
    corners_x = np.array([-x_d/2, -x_d/2, x_d/2, x_d/2], dtype=np.float64)
    corners_y = np.array([-y_d/2, y_d/2, y_d/2, -y_d/2], dtype=np.float64)
    for i in range(4):
        corners[2*i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x
        corners[2*i+1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y


@numba.jit(nopython=True)
def _trangle_area_cpu(a0, a1, b0, b1, c0, c1):
    return ((a0 - c0) * (b1 - c1) - (a1 - c1) * (b0 - c0)) / 2.0


@numba.jit(nopython=True)
def _area_cpu(int_pts, num_of_inter):
    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(_trangle_area_cpu(
            int_pts[0], int_pts[1],
            int_pts[2*i+2], int_pts[2*i+3],
            int_pts[2*i+4], int_pts[2*i+5]))
    return area_val


@numba.jit(nopython=True)
def _sort_vertex_cpu(int_pts, num_of_inter):
    if num_of_inter > 0:
        center_x = 0.0
        center_y = 0.0
        for i in range(num_of_inter):
            center_x += int_pts[2*i]
            center_y += int_pts[2*i+1]
        center_x /= num_of_inter
        center_y /= num_of_inter
        vs = np.zeros(num_of_inter, dtype=np.float64)
        for i in range(num_of_inter):
            vx = int_pts[2*i] - center_x
            vy = int_pts[2*i+1] - center_y
            d = math.sqrt(vx*vx + vy*vy)
            vx = vx / max(d, 1e-8)
            vy = vy / max(d, 1e-8)
            if vy < 0:
                vx = -2 - vx
            vs[i] = vx
        # insertion sort
        for i in range(1, num_of_inter):
            if vs[i-1] > vs[i]:
                temp = vs[i]
                tx = int_pts[2*i]
                ty = int_pts[2*i+1]
                j = i
                while j > 0 and vs[j-1] > temp:
                    vs[j] = vs[j-1]
                    int_pts[j*2] = int_pts[j*2-2]
                    int_pts[j*2+1] = int_pts[j*2-1]
                    j -= 1
                vs[j] = temp
                int_pts[j*2] = tx
                int_pts[j*2+1] = ty


@numba.jit(nopython=True)
def _line_segment_intersection_cpu(pts1, pts2, i, j):
    A0 = pts1[2*i]; A1 = pts1[2*i+1]
    B0 = pts1[2*((i+1)%4)]; B1 = pts1[2*((i+1)%4)+1]
    C0 = pts2[2*j]; C1 = pts2[2*j+1]
    D0 = pts2[2*((j+1)%4)]; D1 = pts2[2*((j+1)%4)+1]
    BA0 = B0 - A0; BA1 = B1 - A1
    DA0 = D0 - A0; CA0 = C0 - A0
    DA1 = D1 - A1; CA1 = C1 - A1
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D1-B1)*(C0-B0) > (C1-B1)*(D0-B0)
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        if abc != abd:
            DC0 = D0-C0; DC1 = D1-C1
            ABBA = A0*B1-B0*A1; CDDC = C0*D1-D0*C1
            DH = BA1*DC0-BA0*DC1
            Dx = ABBA*DC0-BA0*CDDC
            Dy = ABBA*DC1-BA1*CDDC
            return True, Dx/DH, Dy/DH
    return False, 0.0, 0.0


@numba.jit(nopython=True)
def _point_in_quad_cpu(pt_x, pt_y, corners):
    ab0 = corners[2]-corners[0]; ab1 = corners[3]-corners[1]
    ad0 = corners[6]-corners[0]; ad1 = corners[7]-corners[1]
    ap0 = pt_x-corners[0]; ap1 = pt_y-corners[1]
    abab = ab0*ab0+ab1*ab1; abap = ab0*ap0+ab1*ap1
    adad = ad0*ad0+ad1*ad1; adap = ad0*ap0+ad1*ap1
    return abab >= abap and abap >= 0 and adad >= adap and adap >= 0


@numba.jit(nopython=True)
def _single_rotate_iou_cpu(box1, box2, criterion=-1):
    corners1 = np.zeros(8, dtype=np.float64)
    corners2 = np.zeros(8, dtype=np.float64)
    int_pts = np.zeros(16, dtype=np.float64)
    _rbbox_to_corners_cpu(corners1, box1)
    _rbbox_to_corners_cpu(corners2, box2)
    num_of_inter = 0
    for i in range(4):
        if _point_in_quad_cpu(corners1[2*i], corners1[2*i+1], corners2):
            int_pts[num_of_inter*2] = corners1[2*i]
            int_pts[num_of_inter*2+1] = corners1[2*i+1]
            num_of_inter += 1
        if _point_in_quad_cpu(corners2[2*i], corners2[2*i+1], corners1):
            int_pts[num_of_inter*2] = corners2[2*i]
            int_pts[num_of_inter*2+1] = corners2[2*i+1]
            num_of_inter += 1
    for i in range(4):
        for j in range(4):
            has_pt, px, py = _line_segment_intersection_cpu(corners1, corners2, i, j)
            if has_pt:
                int_pts[num_of_inter*2] = px
                int_pts[num_of_inter*2+1] = py
                num_of_inter += 1
    if num_of_inter < 3:
        return 0.0
    _sort_vertex_cpu(int_pts, num_of_inter)
    area_inter = _area_cpu(int_pts, num_of_inter)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    if criterion == -1:
        return area_inter / max(area1 + area2 - area_inter, 1e-8)
    elif criterion == 0:
        return area_inter / max(area1, 1e-8)
    elif criterion == 1:
        return area_inter / max(area2, 1e-8)
    else:
        return area_inter


@numba.jit(nopython=True)
def rotate_iou_cpu_eval(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    iou = np.zeros((N, K), dtype=np.float64)
    for i in range(N):
        for j in range(K):
            iou[i, j] = _single_rotate_iou_cpu(boxes[i], query_boxes[j], criterion)
    return iou


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """Rotated box iou - CPU implementation (avoids numba.cuda segfault)."""
    boxes = boxes.astype(np.float64)
    query_boxes = query_boxes.astype(np.float64)
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    if N == 0 or K == 0:
        return np.zeros((N, K), dtype=np.float64)
    return rotate_iou_cpu_eval(boxes, query_boxes, criterion)
