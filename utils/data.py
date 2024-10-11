import os
import torch
import shutil

import atom3.neighbors as nb
import atom3.pair as pair
from atom3.structure import get_ca_pos_from_residues

from utils.deepinteract_utils import convert_input_pdb_files_to_pair, convert_df_to_dgl_graph
from utils.vector_utils import residue_normals, tangent_vectors
from utils.dataset import DIPSDGLDataset, CASPCAPRIDGLDataset, DB5DGLDataset, OtherTestDataset



def build_input_graph(left_pdb, right_pdb, fast=True):
    # Please rewrite these dirs to your own dirs
    psaia_dir = '~/Programs/PSAIA_1.0_source/bin/linux/psa'
    psaia_config = '~/Programs/psaia_config_file_input.txt'
    hhsuite_db = '~/Data/Databases/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt'

    input_id = "Input"
    try:
        shutil.rmtree(os.path.join('utils', 'datasets', input_id))
    except FileNotFoundError:
        pass
    input_dataset_dir = os.path.join('utils', 'datasets', input_id)

    input_pair = convert_input_pdb_files_to_pair(
        left_pdb,
        right_pdb,
        input_dataset_dir,
        psaia_dir,
        psaia_config,
        hhsuite_db,
        fast=fast)

    lb_df = input_pair.df0
    rb_df = input_pair.df1
    nb_fn = nb.build_get_neighbors("non_heavy_res", 6)
    lres, rres = nb_fn(lb_df, rb_df)
    ldf, rdf = lb_df, rb_df
    lpos = get_ca_pos_from_residues(ldf, lres)
    rpos = get_ca_pos_from_residues(rdf, rres)
    pos_idx, neg_idx = pair._get_residue_positions(ldf, lpos, rdf, rpos, False)
    ca0 = lb_df[lb_df['atom_name'] == 'CA'].reset_index()
    ca1 = rb_df[rb_df['atom_name'] == 'CA'].reset_index()
    label = []
    for raw in pos_idx:
        i, j = raw
        n_i = int(ca0[ca0["index"] == i].index.values)
        n_j = int(ca1[ca1["index"] == j].index.values)
        label.append((n_i, n_j))
    knn = 20
    geo_nbrhd_size = 2
    self_loops = True
    # Convert the input DataFrame into its DGLGraph representations, using all atoms to generate geometric features
    graph1 = convert_df_to_dgl_graph(input_pair.df0, left_pdb, knn, geo_nbrhd_size, self_loops)
    graph2 = convert_df_to_dgl_graph(input_pair.df1, right_pdb, knn, geo_nbrhd_size, self_loops)
    data = {
        'graph1': graph1,
        'graph2': graph2,
        'examples': torch.cat([
            torch.IntTensor(label),
            torch.ones((len(label), 1)).int(),
        ], dim=-1),
        # Both 'complex' and 'filepath' are unused during Lightning's predict_step()
        'complex': left_pdb,
        'filepath': left_pdb,
    }
    return data


def get_data(left_pdb, right_pdb, fast=True):
    data = build_input_graph(left_pdb, right_pdb, fast=fast)
    graph1 = data["graph1"]
    graph2 = data["graph2"]

    x1 = graph1.ndata["x"].contiguous()
    x2 = graph2.ndata["x"].contiguous()

    batch1 = torch.zeros(x1.shape[0], dtype=torch.int).contiguous()
    batch2 = torch.zeros(x2.shape[0], dtype=torch.int).contiguous()

    normal1, _ = residue_normals(x1, batch1)
    normal2, _ = residue_normals(x2, batch2)
    uv1 = tangent_vectors(normal1)
    uv2 = tangent_vectors(normal2)
    nuv1 = torch.cat([normal1[:, None, :], uv1], dim=-2)
    nuv2 = torch.cat([normal2[:, None, :], uv2], dim=-2)

    graph1.ndata["nuv"] = nuv1.detach()
    graph2.ndata["nuv"] = nuv2.detach()

    if data['graph1'].num_nodes() < data['graph2'].num_nodes():
        old_data = data['graph1']
        data['graph1'] = data['graph2']
        data['graph2'] = old_data
        data['examples'] = data['examples'][:, [1, 0, 2]]

    return data


def get_dataset(dataset_name, raw_dir, logging, nuv=False, nuv_angle=False, protrans=False, esm2=False, paired=False):
    if dataset_name == "dips":
        logging.info(f"Protein Protein Docking from DIPS-PLUS by DeepInteract")
        train = DIPSDGLDataset(raw_dir=raw_dir, mode="train", nuv_residue=nuv, nuv_angle=nuv_angle, protrans=protrans, esm2=esm2)
        valid = DIPSDGLDataset(raw_dir=raw_dir, mode="val", nuv_residue=nuv, nuv_angle=nuv_angle, protrans=protrans, esm2=esm2)
        test = DIPSDGLDataset(raw_dir=raw_dir, mode="test", nuv_residue=nuv, nuv_angle=nuv_angle, protrans=protrans, esm2=esm2)
        all_pocket_test = None
        info = None
        return train, None, valid, test, all_pocket_test, info
    elif dataset_name == "casp":
        train = None
        valid = None
        test = CASPCAPRIDGLDataset(raw_dir=raw_dir, mode="test", nuv_residue=nuv, nuv_angle=nuv_angle)
        all_pocket_test = None
        info = None
        return train, None, valid, test, all_pocket_test, info
    elif dataset_name[:6] == "casp15":
        train = None
        valid = None
        test = OtherTestDataset(raw_dir=raw_dir, mode="test", nuv_residue=nuv, nuv_angle=nuv_angle, task=dataset_name)
        all_pocket_test = None
        info = None
        return train, None, valid, test, all_pocket_test, info
    elif dataset_name == "db5":
        train = DB5DGLDataset(raw_dir=raw_dir, mode="train", nuv_residue=nuv, nuv_angle=nuv_angle)
        valid = DB5DGLDataset(raw_dir=raw_dir, mode="val", nuv_residue=nuv, nuv_angle=nuv_angle)
        test = DB5DGLDataset(raw_dir=raw_dir, mode="test", nuv_residue=nuv, nuv_angle=nuv_angle)
        all_pocket_test = None
        info = None
        return train, None, valid, test, all_pocket_test, info
    elif dataset_name == "antibody":
        train = OtherTestDataset(raw_dir=raw_dir, mode="train", nuv_residue=nuv, nuv_angle=nuv_angle, task=dataset_name)
        valid = OtherTestDataset(raw_dir=raw_dir, mode="val", nuv_residue=nuv, nuv_angle=nuv_angle, task=dataset_name)
        test = OtherTestDataset(raw_dir=raw_dir, mode="test", nuv_residue=nuv, nuv_angle=nuv_angle, task=dataset_name)
        all_pocket_test = None
        info = None
        return train, None, valid, test, all_pocket_test, info
    elif dataset_name == "db5.5":
        train = OtherTestDataset(raw_dir=raw_dir, mode="train", nuv_residue=nuv, nuv_angle=nuv_angle, task=dataset_name)
        valid = OtherTestDataset(raw_dir=raw_dir, mode="val", nuv_residue=nuv, nuv_angle=nuv_angle, task=dataset_name)
        test = OtherTestDataset(raw_dir=raw_dir, mode="test", nuv_residue=nuv, nuv_angle=nuv_angle, task=dataset_name)
        all_pocket_test = None
        info = None
        return train, None, valid, test, all_pocket_test, info
    elif dataset_name == "db5.5_all":
        train = None
        valid = None
        test = OtherTestDataset(raw_dir=raw_dir, mode="all", nuv_residue=nuv, nuv_angle=nuv_angle, task=dataset_name)
        all_pocket_test = None
        info = None
        return train, None, valid, test, all_pocket_test, info
    elif dataset_name == "timesplits":
        train = OtherTestDataset(raw_dir=raw_dir, mode="train", nuv_residue=nuv, nuv_angle=nuv_angle, task=dataset_name, esm2=esm2, paired=paired)
        valid = OtherTestDataset(raw_dir=raw_dir, mode="val", nuv_residue=nuv, nuv_angle=nuv_angle, task=dataset_name, esm2=esm2, paired=paired)
        test = OtherTestDataset(raw_dir=raw_dir, mode="test", nuv_residue=nuv, nuv_angle=nuv_angle, task=dataset_name, esm2=esm2, paired=paired)
        all_pocket_test = None
        info = None
        return train, None, valid, test, all_pocket_test, info
    else: 
        raise NotImplementedError()