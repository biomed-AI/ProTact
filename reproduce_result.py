import argparse
import torch

from tqdm import tqdm

from model import ProTact
from utils.deepinteract_utils import calculate_top_k_prec, calculate_top_k_recall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the Contact map.")
    parser.add_argument("--dataset", type=str, default="dips", help="Dataset name (dips, casp, antibody)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use.")
    parser.add_argument("--model", type=str, default="model/best_dips.pt", help="Model path.")

    args = parser.parse_args()
    top_metric = {
        "top_10_prec": [],
        "top_l_by_10_prec": [],
        "top_l_by_5_prec": [],
        "top_l_recall": [],
        "top_l_by_2_recall": [],
        "top_l_by_5_recall": [],
    }
    device = args.device
    model = ProTact().to(device)

    state_dict = torch.load(args.model, map_location=device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if "node_in_embedding.weight" not in state_dict:
        state_dict.setdefault("node_in_embedding.weight", state_dict["node_origin_in_embedding.weight"])
        state_dict.pop("node_origin_in_embedding.weight")
    model.load_state_dict(state_dict)
    model.eval()
    data_list = torch.load("data/test/{}_test_dataset.pt".format(args.dataset))
    with torch.no_grad():
        for data in tqdm(data_list):
            graph1 = data["graph1"]
            graph2 = data["graph2"]
            label = data["examples"][:, 2]
            l = min(graph1.num_nodes(), graph2.num_nodes())
            y_pred = model(graph1.to(device), graph2.to(device))[0]
            pred = torch.flatten(y_pred, end_dim=-2)
            pred_softmax = torch.softmax(pred, dim=-1)
            sorted_pred_indices = torch.argsort(pred_softmax.detach().cpu()[:, 1], descending=True)
            top_metric["top_10_prec"].append(calculate_top_k_prec(sorted_pred_indices, label, 10))
            top_metric["top_l_by_10_prec"].append(calculate_top_k_prec(sorted_pred_indices, label, (l // 10)))
            top_metric["top_l_by_5_prec"].append(calculate_top_k_prec(sorted_pred_indices, label, (l // 5)))
            top_metric["top_l_recall"].append(calculate_top_k_recall(sorted_pred_indices, label, l))
            top_metric["top_l_by_2_recall"].append(calculate_top_k_recall(sorted_pred_indices, label, (l // 2)))
            top_metric["top_l_by_5_recall"].append(calculate_top_k_recall(sorted_pred_indices, label, (l // 5)))

    print({k: torch.Tensor(v).mean() for k, v in top_metric.items()})
