import csv
import glob
import os
import shutil
import pickle
import random


def rename_old_image_file():
    base_path = 'photos/'
    photo_index = {}
    for file_path in glob.glob('images/*'):
        file_name = os.path.basename(file_path)
        member_sequence, _ = file_name.split('_')
        _, file_extension = file_name.split('.')
        new_path = os.path.join(base_path, str(member_sequence) + file_extension)
        shutil.move(file_name, new_path)
        photo_index[member_sequence] = new_path
    with open('photo.info', 'wb') as f:
        pickle.dump(photo_index, f)


def categorize_image():
    base_path = 'id_dataset/'
    log = open('categorize_image.log', 'w')
    with open('photo.info', 'rb') as f:
        photo_index = pickle.load(f)

    with open('member.csv') as csv_file:
        csv_line = csv.reader(csv_file, delimiter='\t')
        for row in enumerate(csv_line):
            member_seq = row[0]
            nation = row[1]
            card_type = row[2]

            if photo_index.get(member_seq) is None:
                log.write('member id: {} photo not found'.format(member_seq))
                continue

            old_path = photo_index[member_seq]
            file_name = os.path.basename(old_path)
            mode = 'train' if random.randint(1, 10) < 9 else 'val'
            new_path = os.path.join(base_path, mode, str(nation) + '_' + str(card_type), file_name)
            shutil.move(old_path, new_path)


def make_dirs():
    base_path = 'id_dataset'
    train_base_path = os.path.join(base_path, 'train')
    val_base_path = os.path.join(base_path, 'val')
    os.mkdir(train_base_path)
    os.mkdir(val_base_path)
    #connect
    #cursor
    #select
    #for
    nation = row[0]
    card_type = row[1]
    os.makedirs(os.path.join(train_base_path, str(nation) + '_' + str(card_type)))
    os.makedirs(os.path.join(val_base_path, str(nation) + '_' + str(card_type)))


def get_image_size_info():
    from pathlib import Path
    from PIL import Image
    photo_info = {}

    for path in Path('id_dataset').rglob('*'):
        file_name = os.path.basename(path)
        _, file_type = file_name.split('.')
        im = Image.open(path)
        key = file_type + str(im.size)
        if photo_info.get(key) is None:
            photo_info[key] = 0
        else:
            photo_info[key] += 1

