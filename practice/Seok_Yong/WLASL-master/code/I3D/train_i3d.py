import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from torchvision import transforms
import videotransforms

import numpy as np

from configs import Config
from pytorch_i3d import InceptionI3d

# from datasets.nslt_dataset import NSLT as Dataset
from datasets.nslt_dataset import NSLT as Dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('--num_class', type=int)

args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def run(configs,
        mode='rgb',
        root='/ssd/Charades_v1_rgb',
        train_split='charades/charades.json',
        save_model='',
        weights=None):
    """모델 학습 실행

    Args:
        configs (Config): ini 파일에서 불러온 하이퍼 파라미터
        mode (str, optional): 비디오 모드, rgb 또는 flow. Defaults to 'rgb'.
        root (dict), optional): 학습에 사용할 영상이 있는 폴더를 word 키에 저장. Defaults to '/ssd/Charades_v1_rgb'.
        train_split (str, optional): 영상별 train, test 여부를 저장한 json 파일 위치. Defaults to 'charades/charades.json'.
        save_model (str, optional): 모델을 저장할 폴더. Defaults to ''.
        weights (str), optional): 기존에 학습한 모델의 위치. Defaults to None.
    """
    print(configs)

    # setup dataset
    # 데이터를 불러올 때 랜덤하게 크롭하거나 가로축으로 뒤집음
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(), ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    # 영상이 있는 폴더에서 train에 해당하는 영상을 데이터셋으로 불러옴
    dataset = Dataset(train_split, 'train', root, mode, train_transforms)
    # Dataset을 iterable하게 만들어주는 DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=configs.batch_size, shuffle=True, num_workers=0,
                                             pin_memory=True)

    # 위와 같이 DataLoader 생성
    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=2,
                                                 pin_memory=False)

    # 위에서 생성한 DataLoader를 저장한 dict
    dataloaders = {'train': dataloader, 'test': val_dataloader}
    datasets = {'train': dataset, 'test': val_dataset}

    # setup the model
    # 2채널, 3채널 영상을 입력받는 I3D 모델 생성
    # I3D 모델은 사전 학습된 ImageNet 모델 사용
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('weights/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt'))

    # 모델의 output 노드 수를 수어 종류에 맞게 설정
    num_classes = dataset.num_classes
    i3d.replace_logits(num_classes)

    # 사전 학습된 모델을 파라미터로 넣었을 경우 학습할 모델에 적용
    if weights:
        print('loading weights {}'.format(weights))
        i3d.load_state_dict(torch.load(weights))

    # GPU 쓰겠다는거같슴다
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    # 초기 학습률
    lr = configs.init_lr

    # 학습률 조절에 사용할 손실함수
    weight_decay = configs.adam_weight_decay  # 가중치 정형화를 위해 사용
    optimizer = optim.Adam(i3d.parameters(), lr=lr, weight_decay=weight_decay)

    # 몇번 반복할때마다 step을 늘릴지
    num_steps_per_update = configs.update_per_step  # accum gradient
    steps = 0
    epoch = 0

    best_val_score = 0
    # scheduler의 인수로 받는 값이 줄어들지 않으면 학습률을 낮춤
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.3)

    # for epoch in range(num_epochs):
    while steps < configs.max_steps and epoch < 400:
        # 밑에 print가 왜 안나올까요...?
        print('Step {}/{}'.format(steps, configs.max_steps))
        print('-' * 10)

        epoch += 1
        # Each epoch has a training and validation phase
        # 에포크마다 train, test 각각 반복
        for phase in ['train', 'test']:
            collected_vids = []  # ???

            # 모델의 train 여부 설정
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode

            # 변수 초기화
            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            # loss 계산에 사용할 confusion matrix
            confusion_matrix = np.zeros(
                (num_classes, num_classes), dtype=int)

            # Iterate over data.
            # train / test에 해당하는 영상별로 반복
            for data in dataloaders[phase]:
                num_iter += 1  # 반복횟수 count 증가인데 이게 else에 들어가야할것같기도하고그렇습니다

                # 해당 데이터의 len이 1보다 작을 경우 ERROR 출력 후 다음 반복 실행
                if data == -1:  # bracewell does not compile opencv with ffmpeg, strange errors occur resulting in no video loaded
                    print("ERROR!")

                # 데이터에 문제가 없을 경우 실행
                else:
                    # get the inputs
                    # inputs, labels, vid, src = data
                    # 해당 영상 정보 받아옴
                    inputs, labels, vid = data

                    """이 밑은 모르겠슴다!"""
                    # wrap them in Variable
                    inputs = inputs.cuda()
                    t = inputs.size(2)
                    labels = labels.cuda()

                    per_frame_logits = i3d(inputs, pretrained=False)
                    # upsample to input size
                    per_frame_logits = F.upsample(
                        per_frame_logits, t, mode='linear')

                    # compute localization loss
                    loc_loss = F.binary_cross_entropy_with_logits(
                        per_frame_logits, labels)
                    tot_loc_loss += loc_loss.data.item()

                    # 예측 및 정답?
                    predictions = torch.max(per_frame_logits, dim=2)[0]
                    gt = torch.max(labels, dim=2)[0]

                    # compute classification loss (with max-pooling along time B x C x T)
                    cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
                                                                  torch.max(labels, dim=2)[0])
                    tot_cls_loss += cls_loss.data.item()

                    for i in range(per_frame_logits.shape[0]):
                        confusion_matrix[torch.argmax(gt[i]).item(
                        ), torch.argmax(predictions[i]).item()] += 1

                    loss = (0.5 * loc_loss + 0.5 * cls_loss) / \
                        num_steps_per_update
                    tot_loss += loss.data.item()
                    if num_iter == num_steps_per_update // 2:
                        print(epoch, steps, loss.data.item())
                    loss.backward()

                    # step 업데이트 조건 만족할 시
                    if num_iter == num_steps_per_update and phase == 'train':
                        # step 증가 및 optimizer 초기화
                        steps += 1
                        num_iter = 0
                        optimizer.step()
                        optimizer.zero_grad()
                        # lr_sched.step()
                        # steps가 10의 배수일 때마다 중간 결과 출력
                        if steps % 10 == 0:
                            acc = float(np.trace(confusion_matrix)) / \
                                np.sum(confusion_matrix)
                            print(
                                'Epoch {} {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(epoch,
                                                                                                                     phase,
                                                                                                                     tot_loc_loss /
                                                                                                                     (10 * num_steps_per_update),
                                                                                                                     tot_cls_loss /
                                                                                                                     (10 * num_steps_per_update),
                                                                                                                     tot_loss / 10,
                                                                                                                     acc))
                            tot_loss = tot_loc_loss = tot_cls_loss = 0.  # loss 초기화

            # test phase일 경우
            if phase == 'test':
                val_score = float(np.trace(confusion_matrix)
                                  ) / np.sum(confusion_matrix)

                # 모델 성능이 개선되었거나 epoch가 홀수일 경우 모델 저장
                if val_score > best_val_score or epoch % 2 == 0:
                    best_val_score = val_score
                    model_name = save_model + "nslt_" + str(num_classes) + str(epoch) + "_" + str(steps).zfill(
                        6) + '_%3f.pt' % val_score

                    torch.save(i3d.module.state_dict(), model_name)
                    print(model_name)

                print('VALIDATION: {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(phase,
                                                                                                              tot_loc_loss / num_iter,
                                                                                                              tot_cls_loss / num_iter,
                                                                                                              (tot_loss * num_steps_per_update) / num_iter,
                                                                                                              val_score
                                                                                                              ))

                # scheduler의 인수로 받는 값이 줄어들지 않으면 학습률을 낮춤
                scheduler.step(tot_loss * num_steps_per_update / num_iter)


if __name__ == '__main__':
    # WLASL setting
    mode = 'rgb'
    root = {'word': '../../data/WLASL2000'}

    save_model = 'checkpoints/'
    train_split = 'preprocess/nslt_2000.json'

    # weights = 'archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'
    weights = None
    config_file = 'configfiles/asl2000.ini'

    configs = Config(config_file)
    print(root, train_split)
    run(configs=configs, mode=mode, root=root, save_model=save_model,
        train_split=train_split, weights=weights)
