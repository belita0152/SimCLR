import os
import torch
import shutil
import yaml


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # 1. 현재 모델 상태 저장
    torch.save(state, filename)

    # 2. 만약 현재 모델이 최적의 모델이라면, 별도의 파일로 저장
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    # 1. 모델 체크포인트 폴더가 존재하지 않으면 생성
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)

        # 2. config.yml 파일을 생성하고, args를 YAML 형식으로 저장
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


