import os
import torch
import shutil

import atom3.neighbors as nb
import atom3.pair as pair
from atom3.structure import get_ca_pos_from_residues

from utils.DeepInteract_utils import convert_input_pdb_files_to_pair, convert_df_to_dgl_graph
from utils.vector_utils import residue_normals, tangent_vectors


def build_input_graph(left_pdb, right_pdb, fast=True):
    # Please rewrite these dirs to your own dirs
    psaia_dir = '~/Programs/PSAIA_1.0_source/bin/linux/psa'
    psaia_config = 'utils/datasets/builder/psaia_config_file_input.txt'
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

    return data
