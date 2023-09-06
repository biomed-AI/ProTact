import argparse
import torch

from model import ProTact
from utils.data import get_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict the Contact map.")
    parser.add_argument("--left_pdb", type=str, help="Left pdb file path.")
    parser.add_argument("--right_pdb", type=str, help="Right pdb file path.")
    parser.add_argument("--model", type=str, default="model/best_dips.pt", help="Model path.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use.")
    parser.add_argument("--no_fast", action="store_true", help="Use the slow version with the genetic databases.")

    args = parser.parse_args()
    args.fast = not args.no_fast
    device = args.device

    model = ProTact().to(device)

    state_dict = torch.load(args.model, map_location=device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if "node_in_embedding.weight" not in state_dict:
        state_dict.setdefault("node_in_embedding.weight", state_dict["node_origin_in_embedding.weight"])
        state_dict.pop("node_origin_in_embedding.weight")
    model.load_state_dict(state_dict)

    data = get_data(args.left_pdb, args.right_pdb, args.fast)

    graph1 = data["graph1"]
    graph2 = data["graph2"]

    y_pred = model(graph1.to(device), graph2.to(device))

    pred = y_pred[0].contiguous()
    contact_map = torch.softmax(pred, dim=-1)[:, :, 1]

    torch.save(contact_map.cpu(), "test_contact_map.pt")
