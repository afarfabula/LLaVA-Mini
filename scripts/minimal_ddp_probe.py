import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def main() -> None:
    backend = os.environ["DDP_BACKEND"]
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model = torch.nn.Linear(8, 8).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    x = torch.randn(4, 8, device=f"cuda:{local_rank}")
    y = model(x).sum()
    y.backward()
    print(f"ok rank={rank} backend={backend}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
