import os
import torch
import random
import numpy as np
from multiprocessing import cpu_count

from tqdm import tqdm
from utils.deepinteract_utils import calculate_top_k_prec, calculate_top_k_recall

def setCpu(core: int = 8):
    cpu_num = core
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)


def Seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def print_metrics(metrics):
    out_list = []
    for key in metrics:
        out_list.append(f"{key}:{metrics[key]:6.5f}")
    out = ", ".join(out_list)
    return out


def evaluate_PPI(
        data_loader,
        model,
        device,
        metrics,
        criterion,
        test_count=False,
        vis=False,
        save=False,
):
    data_it = tqdm(data_loader)
    top_metric = {
        "top_10_prec": [],
        "top_l_by_10_prec": [],
        "top_l_by_5_prec": [],
        "top_l_recall": [],
        "top_l_by_2_recall": [],
        "top_l_by_5_recall": [],
    }
    model.to(device)
    batch_loss = []
    name = []
    for data in data_it:
        graph1, graph2, labels, files = data
        graph1 = graph1.to(device)
        graph2 = graph2.to(device)
        y_pred = model(graph1, graph2)
        for i, label in enumerate(labels):
            name.append(files[i][3:7])
            l = min(graph1.batch_num_nodes()[i], graph2.batch_num_nodes()[i]).item()
            y = label[:, 2].to(device)
            pred = torch.flatten(y_pred[i], end_dim=-2)
            pred_softmax = torch.softmax(pred, dim=-1)
            loss_i = criterion(pred, y)
            batch_loss.append(loss_i.item())
            # Save to Antibody
            file_name = files[i].split("/")[-1][:4].upper()
            file_name = files[i].split("/")[-1][:6].upper() + files[i].split("/")[-1][7:10].upper()
            for k, v in metrics.items():
                v(pred_softmax.detach().cpu(), y.detach().cpu())
            sorted_pred_indices = torch.argsort(pred_softmax.detach().cpu()[:, 1], descending=True)
            y = y.detach().cpu()
            top_metric["top_10_prec"].append(calculate_top_k_prec(sorted_pred_indices, y, k=10))
            top_metric["top_l_by_10_prec"].append(calculate_top_k_prec(sorted_pred_indices, y, k=(l // 10)))
            top_metric["top_l_by_5_prec"].append(calculate_top_k_prec(sorted_pred_indices, y, k=(l // 5)))
            top_metric["top_l_recall"].append(calculate_top_k_recall(sorted_pred_indices, y, k=l))
            top_metric["top_l_by_2_recall"].append(calculate_top_k_recall(sorted_pred_indices, y, k=(l // 2)))
            top_metric["top_l_by_5_recall"].append(calculate_top_k_recall(sorted_pred_indices, y, k=(l // 5)))

            data_it.set_description(f"{loss_i.item():.3}")
    return_metrics = {k: v.compute()[1] for k, v in metrics.items()}
    if test_count:
        return_metrics.update({k: "{}({})".format(torch.mean(torch.Tensor(v)).item(), torch.std(torch.Tensor(v)).item())
                               for k, v in top_metric.items()})
        return_metrics.update({"{}_list".format(k): v
                               for k, v in top_metric.items()})
        return_metrics["name"] = name
    else:
        return_metrics.update({k: torch.mean(torch.Tensor(v))
                               for k, v in top_metric.items()})
    return_metrics.update({"loss": torch.Tensor(batch_loss).mean()})
    return return_metrics
