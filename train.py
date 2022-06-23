# -*- coding: utf-8 -*-
from __future__ import print_function, division
import time
import os
import glob
import argparse
import shutil
from tqdm import tqdm
from PIL import Image

import torch
from torch.optim import lr_scheduler, SGD
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from Model import BasicClassifier
from util import save_checkpoint


def load_dataset(args, mode):
    if mode == 'val':
        data_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.RandomRotation((-90, 90)),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    dataset = datasets.ImageFolder(os.path.join(args.data_dir, mode), data_transforms)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=16)
    data_set_size = len(dataset)

    return dataloader, data_set_size


def test(model, args):
    model.eval()
    data_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for i in range(args.num_class):
        output_path = os.path.join(args.result_dir, str(i))
        if not os.path.exists(output_path):
            os.mkdir(output_path)

    for file_path in glob.glob(args.test_dir + '/*.jpg'):
        file_name = os.path.basename(file_path)
        im = Image.open(file_path)
        im = im.convert('RGB')
        inputs = data_transforms(im)
        inputs = inputs.unsqueeze(0)
        inputs = inputs.cuda()
        with torch.no_grad():
            outputs, _ = model(inputs)
            confidences, predictions = torch.max(outputs, 1)
        prediction = predictions[0].item()
        confidence = confidences[0].item()
        if confidence > 0.5:
            output_path = os.path.join(args.result_dir, str(prediction), file_name)
        else:
            output_path = os.path.join(args.result_dir, file_name)
        shutil.copy(file_path, output_path)
    return None


def train(model, args):
    since = time.time()
    dataloader, data_set_size = load_dataset(args, mode='train')
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=80000, gamma=0.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, verbose=True, factor=0.5)
    checkpoint = None

    max_accuracy = 0.0
    early_stop_count = 40
    for epoch in range(args.n_epoch):
        running_loss = 0.0
        running_corrects = 0
        model.train()

        for step, (inputs, labels) in tqdm(enumerate(dataloader)):
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs, loss = model(inputs, labels)
            loss = loss.sum()
            loss.backward()
            optimizer.step()
            _, predictions = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            corrects = torch.eq(predictions, labels)
            running_corrects += torch.sum(corrects.double())
        epoch_loss = running_loss / data_set_size
        scheduler.step(epoch_loss)
        epoch_acc = float(running_corrects) / data_set_size

        print('Epoc: {} Loss: {:.4f} Acc: {:.4f}, lr:{}'.format(epoch, epoch_loss,
                                                                epoch_acc, optimizer.param_groups[0]['lr']))
        accuracy = evaluate(model, args, epoch, epoch > 10)
        if max_accuracy < accuracy:
            checkpoint = save_checkpoint(model, args.output_dir, epoch)
            max_accuracy = accuracy
        else:
            early_stop_count -= 1

        if early_stop_count < 0:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return checkpoint


def evaluate(model, args, epoch, visualize=False):
    since = time.time()
    model.eval()
    running_corrects = 0
    dataloader, data_set_size = load_dataset(args, mode='val')
    for step, (inputs, labels) in tqdm(enumerate(dataloader)):
        inputs = inputs.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            outputs, _ = model(inputs, labels)
            _, predictions = torch.max(outputs, 1)
            corrects = torch.eq(predictions, labels)
            running_corrects += torch.sum(corrects.double())

        if visualize:
            output_path = os.path.join(args.output_dir, str(epoch))
            if not os.path.exists(output_path):
                os.mkdir(output_path)

            for index, image_tensor in enumerate(inputs):
                if predictions[index] != labels[index]:
                    file_name = str(step) + '_' + str(index) + 'prediction_' + str(predictions[index].item()) + \
                                '_gt_' + str(labels[index].item()) + '.jpg'
                    save_image(image_tensor, os.path.join(output_path, file_name))

    acc = float(running_corrects) / data_set_size
    print('Evaluation Acc: {:.4f}'.format(acc))

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--test_dir', type=str, default='/home/embian/Workspace/data/images/val/ALIEN_REGISTRATION/')
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--data_dir", type=str, default="/home/embian/Workspace/data/images/Classification")
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--result_dir', type=str, default='result/')
    parser.add_argument('--from_checkpoint', type=str, default='./output/15-54-21-72_model.bin')
    # parser.add_argument('--from_checkpoint', type=str)
    parser.add_argument('--num_class', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.0001)

    main_args = parser.parse_args()
    if main_args.from_checkpoint:
        main_model = torch.nn.DataParallel(BasicClassifier(main_args.num_class, hidden=512, freeze_head=False))
        main_model.load_state_dict(torch.load(main_args.from_checkpoint))
    else:
        main_model = torch.nn.DataParallel(BasicClassifier(main_args.num_class))
    main_model.cuda()

    if main_args.do_train:
        check_point = train(main_model, main_args)
        main_model.load_state_dict(torch.load(check_point))

    if main_args.do_eval:
        evaluate(main_model, main_args, main_args.n_epoch, visualize=True)

    if main_args.do_test:
        test(main_model, main_args)
