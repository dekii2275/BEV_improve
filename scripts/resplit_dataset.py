"""
Phân chia lại dataset DAIR-V2X-I cho BEVHeight.
Chỉ sử dụng 8,577 sample có đầy đủ ảnh + label + calib.
Stratified split 80/20 theo dominant object class để đảm bảo
cân bằng phân bố giữa train và val.
"""
import os
import json
import random
from collections import defaultdict

random.seed(42)

# ====== CONFIG ======
DATA_ROOT = "/home/dekii2275/data1/DAIR-V2X-Dataset/infrastructure-side"
OUTPUT_SPLIT = "data/single-infrastructure-split-data.json"
OUTPUT_SPLIT_BACKUP = "data/split-original-backup.json"
TRAIN_RATIO = 0.8

# ====== 1. Tìm sample có đầy đủ data ======
print("=" * 60)
print("DAIR-V2X-I Dataset Re-split Tool")
print("=" * 60)

imgs = set(f.replace('.jpg', '') for f in os.listdir(os.path.join(DATA_ROOT, 'image')) if f.endswith('.jpg'))
labels = set(f.replace('.json', '') for f in os.listdir(os.path.join(DATA_ROOT, 'label/camera')) if f.endswith('.json'))
calib_intr = set(f.replace('.json', '') for f in os.listdir(os.path.join(DATA_ROOT, 'calib/camera_intrinsic')) if f.endswith('.json'))
calib_l2c = set(f.replace('.json', '') for f in os.listdir(os.path.join(DATA_ROOT, 'calib/virtuallidar_to_camera')) if f.endswith('.json'))

complete = sorted(imgs & labels & calib_intr & calib_l2c)
print(f"\nSample đầy đủ: {len(complete)}")

# ====== 2. Phân tích & gom nhóm theo dominant class ======
print("Đang phân tích phân bố object types...")
sample_info = {}
groups = defaultdict(list)

for sid in complete:
    with open(os.path.join(DATA_ROOT, 'label/camera', sid + '.json')) as f:
        objs = json.load(f)
    type_counts = defaultdict(int)
    for o in objs:
        type_counts[o.get('type', 'unknown').lower()] += 1
    dominant = max(type_counts, key=type_counts.get) if type_counts else 'empty'
    sample_info[sid] = {'types': dict(type_counts), 'dominant': dominant, 'n': len(objs)}
    groups[dominant].append(sid)

# ====== 3. Stratified split ======
train_ids, val_ids = [], []
for cls, ids in sorted(groups.items()):
    random.shuffle(ids)
    n = int(len(ids) * TRAIN_RATIO)
    train_ids.extend(ids[:n])
    val_ids.extend(ids[n:])

train_ids.sort()
val_ids.sort()

# ====== 4. Thống kê ======
print(f"\n{'=' * 60}")
print(f"Train: {len(train_ids)} ({len(train_ids)/len(complete)*100:.1f}%)")
print(f"Val:   {len(val_ids)} ({len(val_ids)/len(complete)*100:.1f}%)")

for name, ids in [("TRAIN", train_ids), ("VAL", val_ids)]:
    totals = defaultdict(int)
    for sid in ids:
        for t, c in sample_info[sid]['types'].items():
            totals[t] += c
    print(f"\n{name} objects:")
    for t, c in sorted(totals.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")

# ====== 5. Lưu file ======
if os.path.exists(OUTPUT_SPLIT) and not os.path.exists(OUTPUT_SPLIT_BACKUP):
    import shutil
    shutil.copy2(OUTPUT_SPLIT, OUTPUT_SPLIT_BACKUP)
    print(f"\n✅ Backup split cũ: {OUTPUT_SPLIT_BACKUP}")

with open(OUTPUT_SPLIT, 'w') as f:
    json.dump({'train': train_ids, 'val': val_ids}, f, indent=2)
print(f"✅ Split mới: {OUTPUT_SPLIT}")
