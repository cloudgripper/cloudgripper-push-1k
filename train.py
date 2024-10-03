import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset
from thop import profile
import wandb
import os
from tqdm import tqdm
from VOT import MaxViT, MaxViT_v2, VTN, VTN_MaxViT_v2, SwinTransformer, VTN_Swin
from data_loader import *
import traceback
import utils
import seaborn


os.environ["WANDB_DIR"] = "./wandb"


parser = argparse.ArgumentParser()
parser.add_argument('--vit_architecture', type=str, default="maxvit")
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--data_paths', type=str, nargs='+', default='')
parser.add_argument('--eval_paths', type=str, nargs='+', default='')
parser.add_argument('--test_paths', type=str, nargs='+', default='')
parser.add_argument('--ckpt_path', type=str, default='')
args = parser.parse_args()

config = dict(
    epochs=args.epochs,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    architecture="Transformer",
)
if args.vit_architecture == "maxvit":
    vit = MaxViT(
        num_classes = 1000,
        dim_conv_stem = 64,
        dim = 96,
        dim_head = 32,
        depth = (1, 1, 1, 1),
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1
    )
    vot = VTN(
        vit = vit,
        depth = 2, 
        heads = 8,
        dim_head = 64,
        min_match_ratio=0.9,
        max_distance=0.5
    )
elif args.vit_architecture == "maxvit_2":
    vit = MaxViT_v2(
        num_classes = 1000,
        dim_conv_stem = 64,
        dim = 96,
        dim_head = 32,
        depth = (1, 1, 1, 1),
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1
    )
    vot = VTN(
        vit = vit,
        depth = 2, 
        heads = 8,
        dim_head = 64,
        min_match_ratio=0.9,
        max_distance=0.5
    )
else:  
    vit = SwinTransformer(
        hidden_dim=96,
        layers=(2, 2, 6, 2),
        heads=(3, 6, 12, 24),
        channels=3,
        num_classes=1000,
        head_dim=32,
        window_size=7,
        downscaling_factors=(4, 2, 2, 2),
        relative_pos_embedding=True
    )
    vot = VTN_Swin(
        vit = vit,
        depth = 2, 
        heads = 8,
        dim_head = 64,
        min_match_ratio=0.9,
        max_distance=0.5
    )




def train_batch(data, model, optimizer, criterion, example_ct):
    videos, positions = data
    
    if torch.cuda.is_available():
        videos = videos.to('cuda')
        positions = positions.to('cuda')

    out = model(videos)
    out = out * 255
    positions = positions * 255
    loss = criterion(out, positions)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def setup_run(hyperparameters):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if local_rank == 0:
        run = wandb.init(
            project="OccluManip", config=hyperparameters
        )
    else:
        run = None

    return run

def evaluate_model(model, test_dataloader, device):
    model.eval()
    criterion = nn.MSELoss().to(device)
    with torch.no_grad():
        eval_loss = 0
        for _, data in enumerate(test_dataloader):
            videos, positions = data
            if torch.cuda.is_available():
                videos = videos.to(device)
                positions = positions.to(device)
            out = model(videos)
            out = out * 255
            positions = positions * 255
            loss = criterion(out, positions)
            eval_loss += loss.item()
        eval_loss = eval_loss / len(test_dataloader)
    model.train()
    return eval_loss

def train(run=None):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    is_master = local_rank == 0
    total_devices = torch.cuda.device_count()
    device = torch.device(local_rank % total_devices)
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(device)

    if torch.cuda.is_available():
        model = vot.to(device)
        # print("number of parameters", sum(p.numel() for p in model.parameters()))
        ckpt_path = None
        if dist.get_rank() == 0 and ckpt_path is not None:
            model.load_state_dict(torch.load(ckpt_path))
        model = DDP(model, find_unused_parameters=True, device_ids=[device], output_device=device)
    batch_size = config['batch_size']
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    if is_master:
        run.watch(model)
    train_datasets = []
    for path in args.data_paths:
        train_datasets.append(tensor_loader(path))
    train_dataset = ConcatDataset(train_datasets)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size, sampler=train_sampler)

    test_datasets = []
    for path in args.test_paths:
        test_datasets.append(tensor_loader(path))
    test_dataset = ConcatDataset(test_datasets)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)


    total_step = len(train_dataloader)

    total_batches = len(train_dataloader) * config['epochs']
    example_ct = 0
    batch_ct = 0
    running_loss = 0.0
    for epoch in tqdm(range(config['epochs'])):
        train_dataloader.sampler.set_epoch(epoch)
        for _, data in enumerate(train_dataloader):

            loss = train_batch(data, model, optimizer, criterion, example_ct)
            example_ct +=  len(data)
            batch_ct += 1

            running_loss += loss.item()
        avg_loss = running_loss / len(train_dataloader)
        if is_master:
            run.log({"epoch": epoch, "loss": avg_loss}, step=example_ct)
        print(f"Average loss after epoch {epoch}: {avg_loss:.3f}")
        running_loss = 0.0 

        if epoch % 10 == 0:  
            if is_master:  
                eval_loss = evaluate_model(model, test_dataloader, device) 
                run.log({"epoch": epoch, "eval_loss": eval_loss}, step=example_ct)  


        if epoch % 10 == 0:
            if dist.get_rank()==0:
                torch.save(model.state_dict(), args.ckpt_path +  "/%d.ckpt" % epoch)
    
    return model


run = setup_run(config)
model = train(run)
torch.save(model.state_dict(), args.ckpt_path + "/final_model.pth")
print("Model saved.")



