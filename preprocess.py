import os
import shutil
import pickle
import pymssql
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from random import sample, randint


def categorize_image_by_db_id_type(src_path):
    # identifier_type from DB
    # -------------------------------------------------- -----------
    # NULL                                                      8193
    # ALIEN_REGISTRATION                                      106097
    # DOMESTIC_RESIDENCE                                        1610
    # DRIVER_LICENSE                                           15097
    # NATIONAL_ID                                              16602
    # PASSPORT                                                 30789
    # 그러나 실제로는 passport가 가장 많다. DB 데이터의 신뢰도는 거의 0에 가까웠다.

    log = open('categorize_image.log', 'w')
    conn = pymssql.connect(host=r"localhost", database='remittance', charset='utf8', user='SA', password='Embian1001')
    cursor = conn.cursor()
    cursor.execute("select a.member_seq, a.identifier_type, b.extension \
                    from member a , member_id_photo b where a.member_seq = b.member_seq")
    row = cursor.fetchone()
    while row:
        member_seq = str(row[0])
        id_type = str(row[1])
        extension = str(row[2])
        row = cursor.fetchone()
        file_name = str(member_seq) + extension
        old_path = os.path.join(src_path, file_name)
        if not os.path.exists(old_path):
            log.write(old_path + 'is not exists\n')
            continue

        mode = 'train' if randint(1, 10) < 9 else 'val'
        new_file_name = str(member_seq) + '.' + 'jpg'
        new_path = os.path.join(base_path, mode, id_type, new_file_name)
        print(old_path + ':' + new_path)

        if extension == "png":
            img = Image.open(old_path)
            img = img.convert('RGB')
            img.save(new_path)
        else:
            shutil.move(old_path, new_path)
        row = cursor.fetchone()
    log.close()


def extract_image_size_info(src_path):
    # 현재 image의 size와  width, height ratio가 어느 정도 되는지 조사하기 위해
    # 먼저 size 정보만 뽑아서 pickle로 저장한다.
    from pathlib import Path
    from PIL import Image
    from tqdm import tqdm
    photo_info = {}

    for path in tqdm(Path(src_path).rglob('*')):
        file_name = os.path.basename(path)
        _, file_type = file_name.split('.')
        im = Image.open(path)
        key = file_type + str(im.size)
        if photo_info.get(key) is None:
            photo_info[key] = 0
        else:
            photo_info[key] += 1
    print(photo_info)
    with open('image_size_info.bin', 'wb') as f:
        pickle.dump(photo_info, f)


def print_image_count_below(limit_width, limit_height):
    # limit 보다 작은 개수의 image의 개수를 출력하는 함수이다.
    # 결국에는 excel로 statistics를 작성해서 했기 때문에
    # 쓰이지는 않았다.
    count = 0
    with open('image_size_info.bin', 'rb') as f:
        info = pickle.load(f)

    for key_value in info.keys():
        dict_key = key_value.replace('(', ',').replace(')', '')
        extension, width, height = dict_key.split(',')
        if int(width) < limit_width or int(height) < limit_height:
            count += info[key_value]
    print('The number of images that size is below ({},{}): {}', limit_width, limit_height, count)


def print_image_size_statistic():
    with open('extra_data/image_size_info', 'rb') as f:
        stat = pickle.load(f)

    res = sorted(stat.items(), key=lambda x: x[1])
    for key in res:
        print(key)


# 실제 데이터를 보고 잘 보이지 않는 신분증들의 크기의 하한값을 정한 후 특정 크기 이하의 신분증을 deleted로 옮겼다.
# 여기서는 (256, 256)을 사용했는데 이 이하 크기에서도 잘 보이는 신분증도 있었다.
# 물론 이 보다 큰 size에서도 잘 보이지 않는 신분증도 많이 있다.
def delete_small_images(src_path, width_size, height_size):
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


def categorize_by_nations(data_path, db_data_file_path, target_path):
    # 국가 별로 여권 데이터를 DB에 있는 정보를 이용하여 분류하여 여권의 국가별 비중에 맞춰
    # tagging 데이터를 수집하고자 했다.
    # 데이터가 맞지 않는 경우가 있었으나 그나마 상태가 좋았다. (특히 KOR 쪽에 틀린 데이터가 많다.)
    dict_nation = {}
    head = True
    with open(db_data_file_path) as f:
        while True:
            line = f.readline()
            if head:
                head = False
                continue

            if line:
                items = line.strip().split('\t')
                if items[0] != 'ACTIVE':
                    continue
                member_seq = int(items[2])
                nation = items[3]
                dict_nation[member_seq] = nation
            else:
                break

    for (root, dirs, files) in os.walk(data_path):
        for file_name in files:
            src_image_path = os.path.join(root, file_name)
            member_seq = int(file_name.replace('.jpg', ''))
            nation = dict_nation.get(member_seq)
            if nation is not None:
                target_folder = os.path.join(target_path, nation)
                target_image_path = os.path.join(target_path, nation, file_name)
                if not os.path.exists(target_folder):
                    os.mkdir(target_folder)
                shutil.copy(src_image_path, target_image_path)


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


def member_info_query(member_seq, db_data_file_path):
    with open(db_data_file_path) as f:
        line = f.readline()
        headers = line.strip().split('\t')
        while True:
            line = f.readline()
            if line:
                items = line.strip().split('\t')
                if len(items) > 3 and items[2] == member_seq:
                    for i, item in enumerate(items):
                        print('{}:{}'.format(headers[i], items[i]))
                    break
            else:
                print('can not found member_seq')
                break


if __name__ == '__main__':
    base_path = '/media/hjpark/Samsung_T5/images/'
    db_data_path = 'extra_data/member.tsv'

    delete_small_images(base_path, 256, 256)
    categorize_image_by_db_id_type(base_path)

    domestic_residence_path = '/home/embian/Workspace/data/images/OCR_tagging_added/domestic_residence'
    augment_by_crop_images(domestic_residence_path)

    passport_data_path = '/home/embian/Workspace/data/images/OCR_tagging/passport'
    nation_target_data_path = '/home/embian/Workspace/data/images/nation_classification_passport'
    categorize_by_nations(passport_data_path, nation_target_data_path, db_data_path)

    tagging_path = '/home/embian/Workspace/data/images/OCR_tagging_added'
    classification_path = '/home/embian/Workspace/data/images/Classification'
    data_folder_list = os.listdir(tagging_path)
    for folder in data_folder_list:
        split_train_val(os.path.join(tagging_path, folder), os.path.join(classification_path), 2500)

    member_seq_id = '182546'
    member_info_query(member_seq_id, db_data_path)
