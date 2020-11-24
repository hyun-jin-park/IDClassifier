import os
import shutil
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from random import sample


def delete_small_images(src_path, width_size, height_size):
    # 실제 데이터를 보고 잘 보이지 않는 신분증들의 크기의 하한값을 정한 후 특정 크기 이하의 신분증을 deleted로 옮겼다.
    # 여기서는 (256, 256)을 사용했는데 이 이하 크기에서도 잘 보이는 신분증도 있었다.
    # 물론 이 보다 큰 size에서도 잘 보이지 않는 신분증도 많이 있다.

    for path in tqdm(Path(src_path).rglob('*')):
        file_name = os.path.basename(path)
        _, file_type = file_name.split('.')
        im = Image.open(path)
        if im.size[0] < width_size and im.size[height_size] < 256:
            shutil.move(path, os.path.join('deleted', file_name))


def augment_by_crop_images(src_path):
    for txt_file_path in tqdm(Path(src_path).rglob('*.txt')):
        image_path = txt_file_path.replace('.txt', '.jpg')
        image_file_name = os.path.basename(image_path)
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if len(lines) < 1:
            continue
        items = lines[0].strip().split('\t')
        points = [int(x) for x in items[1:]]
        min_x = min(points[0], points[4])
        min_y = min(points[1], points[5])
        max_x = max(points[0], points[4])
        max_y = max(points[1], points[5])

        crop_area = (min_x, min_y, max_x, max_y)
        im = Image.open(image_path)
        cropped_im = im.crop(crop_area)
        cropped_im.save(os.path.join(src_path, 'cropped_' + image_file_name))


def split_train_val(src_path, dst_path, n_of_samples):
    if not os.path.exists(dst_path):
        os.mkdir(classification_path)
    if not os.path.exists(os.path.join(classification_path, 'train')):
        os.mkdir(os.path.join(classification_path, 'train'))
    if not os.path.exists(os.path.join(classification_path, 'val')):
        os.mkdir(os.path.join(classification_path, 'val'))

    file_list = os.listdir(src_path)
    class_name = src_path.split('/')[-1]

    if not os.path.exists(os.path.join(dst_path, 'train', class_name)):
        os.mkdir(os.path.join(dst_path, 'train', class_name))
        os.mkdir(os.path.join(dst_path, 'val', class_name))

    if len(file_list) > n_of_samples:
        file_list = sample(file_list, n_of_samples)

    train_list = sample(file_list, int(len(file_list)*0.8))

    for path in file_list:
        if path in train_list:
            shutil.copy(os.path.join(src_path, path), os.path.join(dst_path, 'train', class_name, path))
        else:
            shutil.copy(os.path.join(src_path, path), os.path.join(dst_path, 'val', class_name, path))


if __name__ == '__main__':
    base_path = '/media/hjpark/Samsung_T5/images/'
    db_data_path = '/home/embian/Workspace/hanpass/extra_data/member.tsv'
    delete_small_images(base_path, 256, 256)

    domestic_residence_path = '/home/embian/Workspace/data/images/OCR_tagging_added/domestic_residence'
    augment_by_crop_images(domestic_residence_path)

    tagging_path = '/home/embian/Workspace/data/images/OCR_tagging_added'
    classification_path = '/home/embian/Workspace/data/images/Classification'
    data_folder_list = os.listdir(tagging_path)
    for folder in data_folder_list:
        split_train_val(os.path.join(tagging_path, folder), os.path.join(classification_path), 2500)
