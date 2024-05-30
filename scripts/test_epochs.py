import os
import time
import argparse
from mpi4py import MPI


parser = argparse.ArgumentParser(description='multi-gpu test all epochs')
# Configs for testing all checkpoints
parser.add_argument('--interval', type=int, default=25, help='checkpoint saving interval')
parser.add_argument('--start_epoch', type=int, default=325, help='the starting checkpoint epoch to test')
parser.add_argument('--end_epoch', type=int, default=350, help='the ending checkpoint epoch to test')
parser.add_argument('--dataset', default='LaSOT', type=str, help='benchmark to test')
# Configs for multi-GPU testing
parser.add_argument('--threads', default=2, type=int, required=True, help='threads to test checkpoints')
parser.add_argument('--gpus', default='0,1', type=str, required=True, help='gpus')
# Path info
parser.add_argument('--proj_path', type=str, default='ltr/transt/transt', help='project checkpoint path')
parser.add_argument('--arch', type=str, default='transt', help='network architecture')
parser.add_argument('--ts_tag', type=str, default='t', help='t/s, using teacher or student for inference')
args = parser.parse_args()

# Init gpu and epochs
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
gpu_list = args.gpus.split(",")
gpu_nums = len(gpu_list)
GPU_ID = int(gpu_list[0]) + rank % gpu_nums
num_total_tests = (args.end_epoch - args.start_epoch) // args.interval + 1
num_test_each_thread = num_total_tests // args.threads + 1

# Get the name of the node
node_name = MPI.Get_processor_name()
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
print("Node name: {}, GPU_ID: {}".format(node_name, GPU_ID))
time.sleep(rank * 5)

# Run test scripts -- two epochs for each process
for i in range(num_test_each_thread):
    dataset = args.dataset
    try:
        epoch_ID += (args.threads * args.interval)
    except:
        epoch_ID = rank * args.interval + args.start_epoch

    if epoch_ID > args.end_epoch:
        continue

    tag_dict = {'transt': 'TransT', 'ocean': 'Ocean'}
    tag = tag_dict[args.arch]
    resume = 'checkpoints/{}/{}_ep{:04d}_{}.pth.tar'.format(args.proj_path, tag, epoch_ID, args.ts_tag)
    name = '{}_ep{:04d}_{}'.format(tag, epoch_ID, args.ts_tag)
    print('==> test {} th epoch'.format(epoch_ID))
    os.system('python -u ./scripts/test.py --dataset {0} --resume {1} --name {2} --arch {3}'.
              format(dataset, resume, name, args.arch))
