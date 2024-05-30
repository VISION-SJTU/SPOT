import os
import yaml
import argparse
from os.path import exists


def parse_args():
    """
    args for onekey.
    """
    parser = argparse.ArgumentParser(description='Train, Test and eval SPOT with onekey')

    # Train, test and eval tags
    parser.add_argument('--train', action='store_true', help='whether to train')
    parser.add_argument('--test', action='store_true', help='whether to test')
    parser.add_argument('--eval', action='store_true', help='whether to eval')
    # Configs for testing all checkpoints
    parser.add_argument('--interval', type=int, default=25, help='checkpoint saving interval')
    parser.add_argument('--start_epoch', type=int, default=225, help='the starting checkpoint epoch to test')
    parser.add_argument('--end_epoch', type=int, default=300, help='the ending checkpoint epoch to test')
    parser.add_argument('--dataset', type=str, default='GOT-10k', help='benchmark to test')
    # Configs for multi-GPU testing
    parser.add_argument('--gpus', type=str, default="0,1,2,3", help='gpus')
    parser.add_argument('--threads', type=int, default=4, help="threads to conduct multi-gpu testing")
    # Path info
    parser.add_argument('--module', type=str, default='spot', help='project module path')
    parser.add_argument('--arch', type=str, default='transt', help='network architecture')
    parser.add_argument('--variant', type=str, default='spot_transt_003', help='network variant')
    parser.add_argument('--ts_tag', type=str, default='t', help='t/s, using teacher or student for inference')
    args = parser.parse_args()

    # Generating auxiliary arguments
    args.variant = args.arch if args.variant == '' else args.variant
    args.proj_path = 'ltr/{}/{}'.format(args.module, args.variant)
    prefix_dict = {"ocean": "Ocean", "transt": "TransT"}
    args.prefix = prefix_dict[args.arch]
    return args


def main():
    args = parse_args()

    # Training phase -- by default, train 350-400 epochs
    if args.train:
        print('==> Train Phase')
        print('python -u ./scripts/run_training.py {} {}'.format(args.module, args.variant))

        os.system('python -u ./scripts/run_training.py {} {}'.format(args.module, args.variant))

    # Testing phase -- by default, test SP-TransT from ep_325 to ep_400, with epoch interval as 25
    #                              test SP-Ocean from ep_275 to ep_350, with epoch interval as 25
    if args.test:
        print('==> Test Phase')
        print('mpiexec -n {0} python -u ./scripts/test_epochs.py --start_epoch {1} --end_epoch {2} \
                  --interval {3} --threads {0} --dataset {4} --gpus {5} \
                  --proj_path {6} --arch {7} --ts_tag {8}'.format(args.threads,
                  args.start_epoch, args.end_epoch, args.interval, args.dataset,
                  args.gpus, args.proj_path, args.arch, args.ts_tag))

        os.system('mpiexec -n {0} python -u ./scripts/test_epochs.py --start_epoch {1} --end_epoch {2} \
                  --interval {3} --threads {0} --dataset {4} --gpus {5} \
                  --proj_path {6} --arch {7} --ts_tag {8}'.format(args.threads,
                  args.start_epoch, args.end_epoch, args.interval, args.dataset,
                  args.gpus, args.proj_path, args.arch, args.ts_tag))

    # Eval phase -- evaluate testing results of all designated epochs
    if args.eval:
        print('==> Eval Phase')
        print('python ./scripts/eval.py --tracker_path ./var/results --dataset {0} \
              --tracker_prefix {1}'.format(args.dataset, args.prefix))

        os.system('python ./scripts/eval.py --tracker_path ./var/results --dataset {0} \
              --tracker_prefix {1}'.format(args.dataset, args.prefix))


if __name__ == '__main__':
    main()
