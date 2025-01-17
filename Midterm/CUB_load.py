import tarfile
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# # unzip CUB-200-2011.tgz
# with tarfile.open("CUB_200_2011.tgz", 'r:gz') as tar:
#     tar.extractall(path='CUB_orginal')


original_dir = 'CUB_orginal/CUB_200_2011'
base_dir = 'CUB_200_2011'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

images = pd.read_csv(os.path.join(original_dir, 'images.txt'),
                     sep=' ', names=['img_id', 'file_path'])
labels = pd.read_csv(os.path.join(
    original_dir, 'image_class_labels.txt'), sep=' ', names=['img_id', 'class_id'])
train_test_split_info = pd.read_csv(os.path.join(
    original_dir, 'train_test_split.txt'), sep=' ', names=['img_id', 'is_train'])

data = pd.merge(images, labels, on='img_id')
data = pd.merge(data, train_test_split_info, on='img_id')

test_data = data[data['is_train'] == 0]

for _, row in test_data.iterrows():
    src = os.path.join(original_dir, 'images', row['file_path'])
    dst = os.path.join(test_dir, str(row['class_id']), os.path.basename(src))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)

train_data = data[data['is_train'] == 1]
train_data, val_data = train_test_split(
    train_data, test_size=0.2, stratify=train_data['class_id'])

for _, row in train_data.iterrows():
    src = os.path.join(original_dir, 'images', row['file_path'])
    dst = os.path.join(train_dir, str(row['class_id']), os.path.basename(src))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)

for _, row in val_data.iterrows():
    src = os.path.join(original_dir, 'images', row['file_path'])
    dst = os.path.join(val_dir, str(row['class_id']), os.path.basename(src))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)
