import argparse
import logging
import os
import random
import sys
from datetime import datetime

import torch
import torchmetrics as tm

from torch.utils.data import RandomSampler, WeightedRandomSampler, DataLoader

from model import ProTact
from utils.data import get_dataset
from utils.utils import setCpu, Seed_everything, evaluate_PPI, print_metrics
from utils.deepinteract_utils import dgl_picp_collate

setCpu(16)
Seed_everything(42)

parser = argparse.ArgumentParser(description="Train your own model.")
parser.add_argument("-m", "--mode", type=int, default=0, help="mode specify the model to use.")
parser.add_argument("-d", "--data", type=str, default="dips", help="data specify the data to use.")
parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
parser.add_argument("--sample_n", type=int, default=20000, help="number of samples in one epoch.")
parser.add_argument("--debug", action="store_true")
parser.add_argument(
        "--restart", type=str, default=None, help="continue the training from the model we saved."
)
parser.add_argument(
        "--nuv",
        action="store_true",
)
parser.add_argument(
        "--nuv-angle",
        action="store_true",
)
parser.add_argument(
        "--protrans",
        action="store_true"
)
parser.add_argument(
        "--esm2",
        action="store_true"
)
parser.add_argument(
        "--gnn-type",
        type=str,
        default="geotrans",
)
parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
)
parser.add_argument(
        "--layers",
        type=int,
        default=2,
)
parser.add_argument(
        "--interaction-type",
        type=str,
        default="HGCN",
)
parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
)
parser.add_argument(
        "--eta-min",
        type=float,
        default=1e-8,
)
parser.add_argument(
        "--resultFolder", type=str, default="result/", help="information you want to keep a record."
)
parser.add_argument("--label", type=str, default="", help="information you want to keep a record.")
args = parser.parse_args()

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
DEBUG = args.debug

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("")

logging.info(
        f"""\
{' '.join(sys.argv)}
{timestamp}
{args.label}
{args.layers}
--------------------------------
"""
)

torch.multiprocessing.set_sharing_strategy("file_system")

train, train_after_warm_up, valid, test, all_pocket_test, info = get_dataset(
        args.data, logging, nuv=args.nuv, nuv_angle=args.nuv_angle, esm2=args.esm2
)
logging.info(
        f"data point train: {len(train)},"
        f" train_after_warm_up: {len(train_after_warm_up) if train_after_warm_up is not None else 0},"
        f" valid: {len(valid)},"
        f" test: {len(test)}"
)

num_workers = 0
valid_batch_size = test_batch_size = args.batch_size
train_sampler = RandomSampler(train, replacement=True, num_samples=10)
valid_sampler = RandomSampler(valid, replacement=True, num_samples=10)
test_sampler = RandomSampler(test, replacement=True, num_samples=10)
if DEBUG:
    train_loader = DataLoader(
            train,
            batch_size=args.batch_size,
            # follow_batch=["x", "compound_pair"],
            # sampler=train_sampler,
            shuffle=False,
            pin_memory=False,
            num_workers=num_workers,
            collate_fn=dgl_picp_collate,
    )
    valid_loader = DataLoader(
            valid,
            batch_size=valid_batch_size,
            # follow_batch=["x", "compound_pair"],
            # sampler=valid_sampler,
            shuffle=False,
            pin_memory=False,
            num_workers=num_workers,
            collate_fn=dgl_picp_collate,
    )
    test_loader = DataLoader(
            test,
            batch_size=test_batch_size,
            # follow_batch=["x", "compound_pair"],
            # sampler=test_sampler,
            shuffle=False,
            pin_memory=False,
            num_workers=num_workers,
            collate_fn=dgl_picp_collate,
    )
else:
    train_loader = DataLoader(
            train,
            batch_size=args.batch_size,
            # follow_batch=["x", "compound_pair"],
            # sampler=sampler,
            pin_memory=False,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=dgl_picp_collate,
    )
    valid_loader = DataLoader(
            valid,
            batch_size=valid_batch_size,
            # follow_batch=["x", "compound_pair"],
            shuffle=False,
            pin_memory=False,
            num_workers=num_workers,
            collate_fn=dgl_picp_collate,
    )
    test_loader = DataLoader(
            test,
            batch_size=test_batch_size,
            # follow_batch=["x", "compound_pair"],
            shuffle=False,
            pin_memory=False,
            num_workers=num_workers,
            collate_fn=dgl_picp_collate,
    )

if args.data == "db5":
    train_loader = DataLoader(
            train,
            batch_size=args.batch_size,
            # follow_batch=["x", "compound_pair"],
            # sampler=sampler,
            pin_memory=False,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=dgl_picp_collate,
    )
    valid_loader = DataLoader(
            valid,
            batch_size=valid_batch_size,
            # follow_batch=["x", "compound_pair"],
            shuffle=False,
            pin_memory=False,
            num_workers=num_workers,
            collate_fn=dgl_picp_collate,
    )
    test_loader = DataLoader(
            test,
            batch_size=test_batch_size,
            # follow_batch=["x", "compound_pair"],
            shuffle=False,
            pin_memory=False,
            num_workers=num_workers,
            collate_fn=dgl_picp_collate,
    )

# import model is put here due to an error related to torch.utils.data.ConcatDataset after importing torchdrug.
from tankbind.model import *

device = "cuda"
model = ProTact(
                num_node_input_feats=train.num_node_features if not args.paired else train.num_node_features + 788,
                num_edge_input_feats=train.num_edge_features,
                num_interact_layers=args.layers,
                num_gnn_hidden_channels=args.hidden_dim,
                output_emb=False,
        ).to(device)

if args.restart != 'None':
    state_dict = torch.load(args.restart, map_location=device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if "node_in_embedding.weight" not in state_dict:
        state_dict.setdefault("node_in_embedding.weight", state_dict["node_origin_in_embedding.weight"])
        state_dict.pop("node_origin_in_embedding.weight")
    model.load_state_dict(state_dict)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

num_classes = 2
threshold = 0.5
neg_class_weight = 1.0
pos_class_weight = 5.0
criterion = nn.CrossEntropyLoss()

metrics_list = []
valid_metrics_list = []
test_metrics_list = []

best_seletced = "loss"
best_value = 1
epoch_not_improving = 0

for epoch in range(200):
    train_metric = {
        "train_acc": tm.Accuracy(task="multiclass", num_classes=num_classes, average=None),
        "train_prec": tm.Precision(task="multiclass", num_classes=num_classes, average=None),
        "train_recall": tm.Recall(task="multiclass", num_classes=num_classes, average=None),
    }
    val_metric = {
        "val_acc": tm.Accuracy(task="multiclass", num_classes=num_classes, average=None),
        "val_prec": tm.Precision(task="multiclass", num_classes=num_classes, average=None),
        "val_recall": tm.Recall(task="multiclass", num_classes=num_classes, average=None),
        "val_auroc": tm.AUROC(task="multiclass", num_classes=num_classes, average=None),
        "val_auprc": tm.AveragePrecision(task="multiclass", num_classes=num_classes, average=None),
        "val_f1": tm.F1Score(task="multiclass", num_classes=num_classes, average=None),
    }
    test_metric = {
        "test_acc": tm.Accuracy(task="multiclass", num_classes=num_classes, average=None),
        "test_prec": tm.Precision(task="multiclass", num_classes=num_classes, average=None),
        "test_recall": tm.Recall(task="multiclass", num_classes=num_classes, average=None),
        "test_auroc": tm.AUROC(task="multiclass", num_classes=num_classes, average=None),
        "test_auprc": tm.AveragePrecision(task="multiclass", num_classes=num_classes, average=None),
        "test_f1": tm.F1Score(task="multiclass", num_classes=num_classes, average=None),
    }
    model.train()
    batch_loss = []
    data_it = tqdm(train_loader)
    for data in data_it:
        graph1, graph2, labels, files = data
        # print(files, graph1.num_nodes(), graph2.num_nodes())
        graph1 = graph1.to(device)
        graph2 = graph2.to(device)
        optimizer.zero_grad()
        y_pred = model(graph1, graph2)
        y = labels[0][:, 2].to(device)
        pred = torch.flatten(y_pred[0].contiguous(), end_dim=-2)
        pred_softmax = torch.softmax(pred, dim=-1)
        for k, v in train_metric.items():
            v(pred_softmax.detach().cpu(), y.detach().cpu())
        loss = criterion(pred, y)
        del pred, pred_softmax, y_pred, y
        loss.backward()
        optimizer.step()

        batch_loss.append(loss.item())
        data_it.set_description(f"{loss.item():.5}")
    #scheduler.step()
    metrics = {k: v.compute()[1] for k, v in train_metric.items()}
    metrics.update({"loss": torch.Tensor(batch_loss).mean()})
    logging.info(f"epoch {epoch:<4d}, train, " + print_metrics(metrics))
    metrics_list.append(metrics)
    model.eval()
    with torch.no_grad():
        metrics = evaluate_PPI(
                valid_loader,
                model,
                device,
                val_metric,
                criterion,
        )
    if metrics[best_seletced] >= best_value:
        # not improving. (both metrics say there is no improving)
        epoch_not_improving += 1
        ending_message = f" No improvement +{epoch_not_improving}"
    else:
        epoch_not_improving = 0
        best_value = metrics[best_seletced]
        ending_message = " "
    valid_metrics_list.append(metrics)
    logging.info(f"epoch {epoch:<4d}, valid, " + print_metrics(metrics) + ending_message)

    torch.cuda.empty_cache()
    if args.data != "db5":
        with torch.no_grad():
            metrics = evaluate_PPI(
                    test_loader,
                    model,
                    device,
                    test_metric,
                    criterion,
            )
        test_metrics_list.append(metrics)
        logging.info(f"epoch {epoch:<4d}, test,  " + print_metrics(metrics))

    if not DEBUG:
        saveFileName = f"{pre}/results/single_epoch_{epoch}.pt"

        if epoch % 1 == 0:
            torch.save(model.state_dict(), f"{pre}/models/epoch_{epoch}.pt")
        # torch.save((y, y_pred), f"{pre}/results/epoch_{epoch}.pt")
        if epoch_not_improving > 10:
            # early stop.
            print("early stop")
            break
        torch.cuda.empty_cache()
        os.system(f"cp {timestamp}.log {pre}/")

torch.save((metrics_list, valid_metrics_list, test_metrics_list), f"{pre}/metrics.pt")
