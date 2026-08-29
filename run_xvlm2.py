import os
import argparse
import socket
from contextlib import closing


# =========================
# Cluster / Distributed setup
# =========================

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


MASTER_ADDR = os.environ.get("MASTER_ADDR", "127.0.0.1")
MASTER_PORT = os.environ.get("MASTER_PORT", str(find_free_port()))
NNODES = int(os.environ.get("NNODES", "1"))
NODE_RANK = int(os.environ.get("NODE_RANK", "0"))

# Slurm ???? GPU ??(????)
WORLD_SIZE = int(
    os.environ.get(
        "SLURM_NTASKS",
        os.environ.get("WORLD_SIZE", "1")
    )
)

print("========== Distributed Info ==========")
print("MASTER_ADDR:", MASTER_ADDR)
print("MASTER_PORT:", MASTER_PORT)
print("NNODES:", NNODES)
print("NODE_RANK:", NODE_RANK)
print("WORLD_SIZE:", WORLD_SIZE)
print("=======================================")


# =========================
# Launch builder (CLEAN VERSION)
# =========================

def get_dist_launch(args):
    """
    No CUDA_VISIBLE_DEVICES anymore.
    Fully rely on Slurm / torchrun.
    """

    # ?? / ????
    if args.dist == "auto" or WORLD_SIZE == 1:
        return (
            "torchrun "
            "--standalone "
            "--nproc_per_node=1 "
            f"--master_addr={MASTER_ADDR} "
            f"--master_port={MASTER_PORT} "
        )

    # ?????(??)
    return (
        "torchrun "
        f"--nproc_per_node={WORLD_SIZE} "
        f"--nnodes={NNODES} "
        f"--node_rank={NODE_RANK} "
        f"--master_addr={MASTER_ADDR} "
        f"--master_port={MASTER_PORT} "
    )


# =========================
# Main run
# =========================

def run(args):
    config_path = os.path.join("configs", f"{args.task}.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Please ensure configs/{args.task}.yaml exists."
        )

    print(f"[INFO] Task: {args.task}")
    print(f"[INFO] Config: {config_path}")

    dist_launch = get_dist_launch(args)

    command = (
        f"{dist_launch} Search_xvlm2.py "
        f"--config {config_path} "
        f"--task {args.task} "
        f"--output_dir {args.output_dir} "
        f"--checkpoint {args.checkpoint} "
        f"--bs {args.bs} "
        f"--epo {args.epo} "
        f"--lr {args.lr} "
        f"--seed {args.seed} "
        f"{'--evaluate' if args.evaluate else ''}"
    )

    print("\n========== Launch Command ==========")
    print(command)
    print("====================================\n")

    os.system(command)


# =========================
# Entry
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', default='pab', type=str)
    parser.add_argument('--dist', default='auto', type=str, help="auto = cluster safe mode")

    parser.add_argument('--output_dir', default='output/default_run', type=str)
    parser.add_argument('--checkpoint', default='checkpoint/x2vlm_base_1b.th', type=str)

    parser.add_argument('--bs', default=0, type=int)
    parser.add_argument('--epo', default=0, type=int)
    parser.add_argument('--lr', default=0.0, type=float)
    parser.add_argument('--seed', default=3407, type=int)

    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()
    run(args)