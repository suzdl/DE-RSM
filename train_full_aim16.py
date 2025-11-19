import argparse
from pathlib import Path
from src.src_main_full_aim16 import AbsRel_depth
# from src.src_main_pro import AbsRel_depth
from src.networks import UNet
from src.utils import str2bool, DDPutils
import os
import torch
from torch.backends import cudnn

os.environ["CUDA_VISIBLE_DEVICES"] = '3,4'

# turn fast mode on
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_arguments():
    parser = argparse.ArgumentParser(
        "options for AbsRel_depth estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--save_dir",
        action="store",
        type=str,
        default=r'train_logs_full_aim16_gsrv1_wonoise',
        help="Path to the directory for saving the logs and models",
        required=False
    )
    parser.add_argument(
        "--epochs",
        action="store",
        type=int,
        required=False,
        nargs="+",
        default=60,
        help="epochs numbers",
    )
    parser.add_argument(
        "--batch_size",
        action="store",
        type=int,
        required=False,
        nargs="+",
        default=16,
        help="batch sizes",
    )
    parser.add_argument(
        "--resume_train",
        action="store",
        type=str2bool,
        default=False,
        help="resume train or not",
        required=False,
    )
    # parser.add_argument(
    #     "--model_dir",
    #     action="store",
    #     type=lambda x: Path(x),
    #     default=r'train_logs/models/epoch_80.pth',
    #     help="Path to load models",
    #     required=False
    # )
    parser.add_argument(
        "--port",
        action="store",
        type=int,
        default=6195,
        help="DDP port",
    )

    args = parser.parse_args()
    return args


def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))


def DDP_main(rank, world_size):
    args = parse_arguments()
    args.save_dir = Path(args.save_dir)
    args.model_dir = args.save_dir / 'models' / 'epoch_50.pth'

    # DDP components
    DDPutils.setup(rank, world_size, args.port)

    if rank == 0:
        print(f"Selected arguments: {args}")

    network = UNet()
    print_model_parm_nums(network)

    semigan = AbsRel_depth(network, rank)

    # resume train
    if args.resume_train:
        if rank == 0:
            print('resume training...')
            # load everything
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(args.model_dir, map_location=map_location)
    else:
        checkpoint = None

    semigan.train(
        args=args,
        rank=rank,
        learning_rate=0.0001,
        feedback_factor=1000,
        checkpoint_factor=10,
        num_workers=2,
        checkpoint=checkpoint,
    )

    DDPutils.cleanup()


if __name__ == "__main__":
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        DDPutils.run_demo(DDP_main, n_gpus)

# tensorboard --logdir=train_logs_part_rz_sb_mar_2gpu/logs/tensorboard --port=6090
# /home/szdl/.conda/envs/szdl_3.8/bin/python /home/szdl/suzhangdelong/g2_reproject/train_full_aim16.py