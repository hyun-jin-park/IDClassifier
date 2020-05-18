# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
from torch.optim import lr_scheduler, SGD
from torchvision import datasets, transforms
from Model import BasicClassifier
from torch.utils.data import DataLoader
from util import save_checkpoint
from tqdm import tqdm

import time
import os
import argparse


def load_dataset(args, mode):
    data_transforms = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize((358, 224)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(os.path.join(args.data_dir, mode), data_transforms)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)
    data_set_size = len(dataset)

    return dataloader, data_set_size


def train(model, args):
    since = time.time()
    model.train()

    dataloader, data_set_size = load_dataset(args, mode='train')
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    checkpoint = None

    for epoch in range(args.n_epoch):
        running_loss = 0.0
        running_corrects = 0

        for step, (inputs, labels) in tqdm(enumerate(dataloader)):
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs, loss = model(inputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            _, predictions = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            corrects = torch.eq(predictions, labels)
            running_corrects += torch.sum(corrects.double())
            if step % 10 == 9:
                print('Step: {} Lr: {:.4f} Loss: {:.4f} Acc: {:.4f}'.format(step, optimizer.param_groups[0]['lr'],
                                                                            loss.item(), torch.sum(corrects.double())))
        epoch_loss = running_loss / data_set_size
        epoch_acc = float(running_corrects) / data_set_size

        print('Epoc: {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
        checkpoint = save_checkpoint(model, args.output_dir, epoch)
        evaluate(model, args)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return checkpoint


def evaluate(model, args):
    since = time.time()
    model.eval()
    running_corrects = 0

    dataloader, data_set_size = load_dataset(args, mode='val')
    for step, (inputs, labels) in tqdm(enumerate(dataloader)):
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs, _ = model(inputs, labels)
        _, predictions = torch.max(outputs, 1)
        corrects = torch.eq(predictions, labels)
        running_corrects += torch.sum(corrects.double())
        if step % 10 == 9:
            print('Step: {} Acc: {:.4f}'.format(step, torch.sum(corrects.double())))
    epoch_acc = float(running_corrects) / data_set_size
    print('Evaluation Acc: {:.4f}'.format(epoch_acc))

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--n_epoch',type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--data_dir", type=str, default="./dataset")
    parser.add_argument('--output_dir', type=str, default='./output')
    # parser.add_argument('--from_checkpoint', type=str, default='./output/19-33-48-0_model.bin' )
    parser.add_argument('--from_checkpoint', type=str)
    parser.add_argument('--num_class', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)

    main_args = parser.parse_args()

    main_model = torch.nn.DataParallel(BasicClassifier(main_args.num_class))
    main_model.cuda()

    if main_args.from_checkpoint:
        main_model.load_state_dict(torch.load(main_args.from_checkpoint))

    if main_args.do_train:
        last_checkpoint = train(main_model, main_args)
        main_model.load_state_dict(torch.load(last_checkpoint))

    if main_args.do_eval:
        evaluate(main_model, main_args)