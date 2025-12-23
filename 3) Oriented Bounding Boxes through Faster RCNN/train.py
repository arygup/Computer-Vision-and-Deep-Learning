import torch
import argparse
import os
import numpy as np
import yaml
import random
from tqdm import tqdm
import torchvision
from dataset.st import SceneTextDataset
from torch.utils.data.dataloader import DataLoader

import detection
from detection.faster_rcnn import FastRCNNPredictor
from detection.anchor_utils import AnchorGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_function(data):
    return tuple(zip(*data))


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    dataset_config = config['dataset_params']
    train_config = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    st = SceneTextDataset('train', root_dir=dataset_config['root_dir'])

    train_dataset = DataLoader(st,
                               batch_size=4,
                               shuffle=True,
                               num_workers=4,
                               collate_fn=collate_function)

    faster_rcnn_model = detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                        min_size=600,
                                                        max_size=1000,
    )
    faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(
        faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=dataset_config['num_classes'])

    faster_rcnn_model.train()
    faster_rcnn_model.to(device)
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    optimizer = torch.optim.SGD(lr=1E-4,
                                params=filter(lambda p: p.requires_grad, faster_rcnn_model.parameters()),
                                weight_decay=5E-5, momentum=0.9)

    num_epochs = train_config['num_epochs']
    step_count = 0

    for i in range(num_epochs):
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []
        for ims, targets, _ in tqdm(train_dataset):
            optimizer.zero_grad()
            for target in targets:
                target['boxes'] = target['bboxes'].float().to(device)
                del target['bboxes']
                target['labels'] = target['labels'].long().to(device)
            images = [im.float().to(device) for im in ims]
            batch_losses = faster_rcnn_model(images, targets)
            loss = batch_losses['loss_classifier']
            loss += batch_losses['loss_box_reg']
            loss += batch_losses['loss_rpn_box_reg']
            loss += batch_losses['loss_objectness']

            rpn_classification_losses.append(batch_losses['loss_objectness'].item())
            rpn_localization_losses.append(batch_losses['loss_rpn_box_reg'].item())
            frcnn_classification_losses.append(batch_losses['loss_classifier'].item())
            frcnn_localization_losses.append(batch_losses['loss_box_reg'].item())

            loss.backward()
            optimizer.step()
            step_count +=1
        print('Finished epoch {}'.format(i))
        torch.save(faster_rcnn_model.state_dict(), os.path.join(train_config['task_name'],
                                                                'tv_frcnn_r50fpn_' + train_config['ckpt_name']))
        loss_output = ''
        loss_output += 'RPN Classification Loss : {:.4f}'.format(np.mean(rpn_classification_losses))
        loss_output += ' | RPN Localization Loss : {:.4f}'.format(np.mean(rpn_localization_losses))
        loss_output += ' | FRCNN Classification Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
        loss_output += ' | FRCNN Localization Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))
        print(loss_output)
    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn using torchvision code training')
    parser.add_argument('--config', dest='config_path',
                        default='config/st.yaml', type=str)
    args = parser.parse_args()
    train(args)