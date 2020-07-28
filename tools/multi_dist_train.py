import os
import argparse


parser = argparse.ArgumentParser(description="AML Generic Launcher")
parser.add_argument('--gpu_num_per_node', default="", help="GPU number per node.")
parser.add_argument('--config_file', default="", help="Config file.")
parser.add_argument('--work_dir', default="", help="Work Dir to save models.")
args, _ = parser.parse_known_args()

# start training
os.system("nvidia-smi")
os.system("gcc --version")


# show ip info
#os.system('apt-get install iproute iproute-doc -y')
#os.system('ip addr')


def get_master_container_ip_port():
    return os.getenv('AZ_BATCH_MASTER_NODE').split(':')


if os.environ.get('OMPI_COMM_WORLD_SIZE') is None:
    command_per_node = "export NCCL_IB_DISABLE=1 && export NCCL_SOCKET_IFNAME=eth0 " \
                       "&& python -m torch.distributed.launch --nproc_per_node=%s " \
                       "tools/train.py %s --work_dir %s --validate --launcher pytorch" \
                       % (args.gpu_num_per_node, args.config_file, args.work_dir)
else:
    nnodes = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
    node_rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
    master_ip, master_port = get_master_container_ip_port()
    command_per_node =  "export NCCL_IB_DISABLE=1 && export NCCL_SOCKET_IFNAME=eth0 " \
                        "&& python -m torch.distributed.launch --nproc_per_node=%s " \
                        "--nnodes %d --node_rank %d --master_addr %s --master_port %s " \
                        "tools/train.py %s --work_dir %s --validate --launcher pytorch" \
                        % (args.gpu_num_per_node, nnodes, node_rank, master_ip, master_port, args.config_file, args.work_dir)

print(command_per_node)
os.system(command_per_node)
