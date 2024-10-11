
import logging
import os
import pickle

import Bio
import torch

import numpy as np
import pandas as pd
from tqdm import tqdm
from dgl.data import DGLDataset, download, check_sha1

from atom3.structure import df_to_pdb, get_chain_to_valid_residues
import Bio.PDB.Polypeptide as poly
from Bio.SeqIO import FastaIO
from scipy.spatial.transform import Rotation

from utils.vector_utils import residue_normals, tangent_vectors
from utils.deepinteract_utils \
    import construct_filenames_frame_txt_filenames, build_filenames_frame_error_message, process_complex_into_dict, \
    zero_out_complex_features, dgl_picp_collate


def write_fasta(df0, name, fasta_out):
    df = df0
    pdb = df_to_pdb(df)
    flat_map = {}
    fasta_name_to_chain = {}
    for (chain, residues) in get_chain_to_valid_residues(pdb):
        fasta_name = name + '-' + chain[-2] + '-' + chain[-1]
        flat_map[fasta_name] = residues
        fasta_name_to_chain[fasta_name] = chain
    records = []
    for name, seq in flat_map.items():
        for residue in seq:
            if residue.resname == 'HID':
                residue.resname = 'HIS'
            elif residue.resname == 'CYX':
                residue.resname = 'CYS'
            elif residue.resname == 'ASX':
                residue.resname = 'ASP'
            elif residue.resname == 'GLX':
                residue.resname = 'GLY'
        sequence = [poly.three_to_one(residue.resname) for residue in seq
                    if residue.resname != 'SEC' and residue.resname != 'PYL']
        residue_list = "".join(sequence)
        record = Bio.SeqRecord.SeqRecord(
            Bio.Seq.Seq(residue_list), id=name, description="")
        records.append(record)
    fasta_dir = os.path.dirname(os.path.abspath(fasta_out))
    if not os.path.exists(fasta_dir):
        os.makedirs(fasta_dir, exist_ok=True)
    fasta_out = FastaIO.FastaWriter(open(fasta_out, "w"), wrap=None)
    fasta_out.write_file(records)

def get_sequence(dill_path, dataset):
    complex, df0, df1, complex_id, _, _, c_id, _ = pd.read_pickle(dill_path)
    fasta_name = complex + "_{}".format(c_id)
    write_fasta(df0, fasta_name + "-0", "data/{}/fasta/{}.fasta".format(dataset, fasta_name + "-0"))
    write_fasta(df1, fasta_name + "-1", "data/{}/fasta/{}.fasta".format(dataset, fasta_name + "-1"))


def get_embedding(path_dict: dict):
    chains = list(path_dict.keys())
    chains.sort()
    emb_list = []
    for c in chains:
        emb = torch.from_numpy(np.load(path_dict[c]))
        emb_list.append(emb)
    return torch.cat(emb_list)

#-------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepInteract (https://github.com/BioinfoMachineLearning/DeepInteract):
#-------------------------------------------------------------------------------------------------------------------------------------


class DIPSDGLDataset(DGLDataset):
    r"""Bound protein complex dataset for DGL with PyTorch.

    Statistics:

    - Train examples: 15,618
    - Val examples: 3,548
    - Test examples: 32
    - Number of structures per complex: 2
    ----------------------
    - Total examples: 19,198
    ----------------------

    Parameters
    ----------
    mode: str, optional
        Should be one of ['train', 'val', 'test']. Default: 'train'.
    raw_dir: str
        Raw file directory to download/contains the input data directory. Default: 'final/raw'.
    knn: int
        How many nearest neighbors to which to connect a given node. Default: 20.
    geo_nbrhd_size: int
        Size of each edge's neighborhood when updating geometric edge features. Default: 2.
    self_loops: bool
        Whether to connect a given node to itself. Default: True.
    pn_ratio: float
        The positive-negative ratio to use when assembling training labels for node-node pairs. Default: 0.1.
    percent_to_use: float
        How much of the dataset to load. Default: 1.00.
    process_complexes: bool
        Whether to process each unprocessed complex as we load in the dataset. Default: True.
    input_indep: bool
        Whether to zero-out each input node and edge feature for an input-independent baseline. Default: False.
    train_viz: bool
        Whether to load in complexes to be used for visualizing model training dynamics. Default: False.
    force_reload: bool
        Whether to reload the dataset. Default: False.
    verbose: bool
        Whether to print out progress information. Default: False.

    Notes
    -----
    All the samples will be loaded and preprocessed in the memory first.

    Examples
    --------
    >>> # Get dataset
    >>> train_data = DIPSDGLDataset()
    >>> val_data = DIPSDGLDataset(mode='val')
    >>> test_data = DIPSDGLDataset(mode='test')
    >>>
    >>> len(train_data)
    15618
    >>> train_data.num_chains
    2
    """

    def __init__(self,
                 mode='train',
                 raw_dir="../DIPS/final/raw",
                 knn=20,
                 geo_nbrhd_size=2,
                 self_loops=True,
                 pn_ratio=0.1,
                 percent_to_use=1.00,
                 process_complexes=True,
                 input_indep=False,
                 train_viz=False,
                 force_reload=False,
                 verbose=False,
                 reverse_input=True,
                 nuv_residue=False,
                 nuv_angle=False,
                 protrans=False,
                 esm2=False,
                 generate_fasta=False, ):
        assert mode in ['train', 'val', 'test']
        assert 0.0 < pn_ratio <= 1.0
        assert 0.0 < percent_to_use <= 1.0
        self.mode = mode
        self.root = raw_dir
        self.knn = knn
        self.geo_nbrhd_size = geo_nbrhd_size
        self.reverse_input = reverse_input
        self.self_loops = self_loops
        self.pn_ratio = pn_ratio
        self.percent_to_use = percent_to_use  # How much of the requested dataset (e.g. DIPS-Plus) to use
        self.process_complexes = process_complexes  # Whether to process any unprocessed complexes before training
        self.input_indep = input_indep  # Whether to use an input-independent pipeline to train the model
        self.train_viz = train_viz  # Whether to curate the training loop's validation samples for visualization
        self.final_dir = f"{os.sep}".join(self.root.split(os.sep)[:-1])
        self.processed_dir = os.path.join(self.final_dir, 'processed')
        self.nuv_residue = nuv_residue
        self.nuv_angle = nuv_angle
        self.protrans = protrans
        self.esm2 = esm2
        self.generate_fasta = generate_fasta

        self.filename_sampling = 0.0 < self.percent_to_use < 1.0
        self.base_txt_filename, self.filenames_frame_txt_filename, self.filenames_frame_txt_filepath = \
            construct_filenames_frame_txt_filenames(self.mode, self.percent_to_use, self.filename_sampling, self.root)

        # Try to load the text file containing all DIPS-Plus filenames, and alert the user if it is missing or corrupted
        filenames_frame_to_be_written = not os.path.exists(self.filenames_frame_txt_filepath)
        # Randomly sample DataFrame of filenames with requested cross validation ratio
        if self.filename_sampling:
            if filenames_frame_to_be_written:
                try:
                    self.filenames_frame = pd.read_csv(
                        os.path.join(self.root, self.base_txt_filename + '.txt'), header=None)
                except Exception:
                    raise FileNotFoundError(
                        build_filenames_frame_error_message('DIPS-Plus', 'load', self.filenames_frame_txt_filepath))
                self.filenames_frame = self.filenames_frame.sample(frac=self.percent_to_use).reset_index()
                try:
                    self.filenames_frame[0].to_csv(self.filenames_frame_txt_filepath, header=None, index=None)
                except Exception:
                    raise Exception(
                        build_filenames_frame_error_message('DIPS-Plus', 'write',
                                                            self.filenames_frame_txt_filepath))

        # Load in existing DataFrame of filenames as requested (or if a sampled DataFrame .txt has already been written)
        if not filenames_frame_to_be_written:
            try:
                self.filenames_frame = pd.read_csv(self.filenames_frame_txt_filepath, header=None)
            except Exception:
                raise FileNotFoundError(
                    build_filenames_frame_error_message('DIPS-Plus', 'load', self.filenames_frame_txt_filepath))

        # If requested, filter out all complexes except the first n to reduce memory requirements for our viz samples
        if self.train_viz:
            n = 5532  # Supports up to a world size of 5,532 GPUs (i.e., asserts that n >= total_num_gpus_used)
            self.filenames_frame = self.filenames_frame.head(n=1)
            self.filenames_frame = pd.DataFrame(np.repeat(self.filenames_frame.values, n, axis=0))
            mode = 'viz'

        super(DIPSDGLDataset, self).__init__(name='DIPS-Plus',
                                             raw_dir=raw_dir,
                                             force_reload=force_reload,
                                             verbose=verbose)
        logging.info(f"Loaded DIPS-Plus {mode}-set, source: {self.processed_dir}, length: {len(self)}")
        if self.protrans:
            self.protrans_emb = {}
            for root, _, files in os.walk("DeepInteract/data/DIPS/embedding_protrans"):
                for f in files:
                    cplex, pair, chain_1, chain = f[:-4].split("-")
                    # assert chain_1 == "1"
                    self.protrans_emb.setdefault(cplex, {})
                    self.protrans_emb[cplex].setdefault(int(pair), {})
                    self.protrans_emb[cplex][int(pair)][chain] = os.path.join(root, f)
            drop_index = []
            for (i, raw_path) in self.filenames_frame.iterrows():
                cplex = raw_path[0][3:-5]
                if cplex not in self.protrans_emb:
                    drop_index.append(i)
            logging.info("Loaded prottrans embedding and missing {}".format(len(drop_index)))
            self.filenames_frame = self.filenames_frame.drop(drop_index)

    def download(self):
        """Download and extract a pre-packaged version of the raw pairs if 'self.raw_dir' is not already populated."""
        # Path to store the file
        gz_file_path = os.path.join(os.path.join(*self.raw_dir.split(os.sep)[:-1]), 'final_raw_dips.tar.gz')

        # Download file
        download(self.url, path=gz_file_path)

        # Check SHA-1
        if not check_sha1(gz_file_path, self._sha1_str):
            raise UserWarning('File {} is downloaded but the content hash does not match.'
                              'The repo may be outdated or download may be incomplete. '
                              'Otherwise you can create an issue for it.'.format(gz_file_path))

        # Remove existing raw directory to make way for the new archive to be extracted
        if os.path.exists(self.raw_dir):
            os.removedirs(self.raw_dir)

        # Extract archive to parent directory of `self.raw_dir`
        self._extract_gz(gz_file_path, os.path.join(*self.raw_dir.split(os.sep)[:-1]))

    def process(self):
        """Process each protein complex into a training-validation-ready dictionary representing both structures."""
        if self.process_complexes:
            # Ensure the directory of processed complexes is already created
            os.makedirs(self.processed_dir, exist_ok=True)
            # Process each unprocessed protein complex
            for (i, raw_path) in self.filenames_frame.iterrows():
                raw_filepath = os.path.join(self.raw_dir, f'{os.path.splitext(raw_path[0])[0]}.dill')
                processed_filepath = os.path.join(self.processed_dir, f'{os.path.splitext(raw_path[0])[0]}.dill')
                if not os.path.exists(processed_filepath):
                    processed_parent_dir_to_make = os.path.join(self.processed_dir, os.path.split(raw_path[0])[0])
                    os.makedirs(processed_parent_dir_to_make, exist_ok=True)
                    process_complex_into_dict(raw_filepath, processed_filepath, self.knn,
                                              self.geo_nbrhd_size, self.self_loops, check_sequence=False)

    def has_cache(self):
        """Check if each complex is downloaded and available for training, validation, or testing."""
        for (i, raw_path) in self.filenames_frame.iterrows():
            processed_filepath = os.path.join(self.processed_dir, f'{os.path.splitext(raw_path[0])[0]}.dill')
            if not os.path.exists(processed_filepath):
                logging.info(
                    f'Unable to load at least one processed DIPS-Plus pair. '
                    f'Please make sure all processed pairs have been successfully downloaded and are not corrupted.')
                raise FileNotFoundError
            # TODO: acc
            break
        logging.info('DIPS-Plus cache found')  # Otherwise, a cache was found!

    def __getitem__(self, idx):
        r""" Get feature dictionary by index of complex.

        Parameters
        ----------
        idx : int

        Returns
        -------
        :class:`dict`

    - ``complex['graph1']:`` DGLGraph (of length M) containing each of the first data's encoded node and edge features
    - ``complex['graph2']:`` DGLGraph (of length N) containing each of the second data's encoded node and edge features
    - ``complex['examples']:`` PyTorch Tensor (of shape (M x N) x 3) containing the labels for inter-data node pairs
    - ``complex['complex']:`` Python string describing the complex's code and original pdb filename
    - ``complex['filepath']:`` Python string describing the complex's filepath
        """
        # Assemble filepath of processed protein complex
        complex_filepath = f'{os.path.splitext(self.filenames_frame.iat[idx, 0])[0]}.dill'
        processed_filepath = os.path.join(self.processed_dir, complex_filepath)

        if self.generate_fasta:
            raw_path = self.filenames_frame[0][idx]
            pre_file = "../interim/pairs-pruned/{}".format(
                raw_path)
            if os.path.exists(pre_file):
                get_sequence(pre_file, "DIPS")
            else:
                print(raw_path)

        # Load in processed complex
        with open(processed_filepath, 'rb') as f:
            processed_complex = pickle.load(f)
        processed_complex['filepath'] = complex_filepath  # Add filepath to each complex dictionary

        graph1 = processed_complex["graph1"]
        graph2 = processed_complex["graph2"]

        if self.protrans:
            cplex = complex_filepath[3:-5]
            try:
                graph1_emb = get_embedding(self.protrans_emb[cplex][0])
                graph2_emb = get_embedding(self.protrans_emb[cplex][1])
            except:
                if self.mode != "test":
                    logging.info(f"{cplex} prottrans embedding can't be read!")
                    return self.__getitem__((idx + 1) % len(self))

            if self.mode != "test":
                if graph1.num_nodes() != graph1_emb.shape[0] or graph2.num_nodes() != graph2_emb.shape[0]:
                    logging.info(f"{cplex} prottrans embedding can't match!")
                    return self.__getitem__((idx + 1) % len(self))

            graph1.ndata['f'] = torch.cat([graph1.ndata["f"], graph1_emb], dim=-1)
            graph2.ndata['f'] = torch.cat([graph2.ndata["f"], graph2_emb], dim=-1)

        if self.esm2:
            cplex = complex_filepath[3:-5]
            try:
                # graph1_emb = get_embedding(self.esm2_emb[cplex][0])
                # graph2_emb = get_embedding(self.esm2_emb[cplex][1])
                graph1_emb = graph1.ndata['esm']
                graph2_emb = graph2.ndata['esm']
            except:
                if self.mode != "test":
                    logging.info(f"{cplex} esm2 embedding can't be read!")
                    return self.__getitem__((idx + 1) % len(self))

            if self.mode != "test":
                if graph1.num_nodes() != graph1_emb.shape[0] or graph2.num_nodes() != graph2_emb.shape[0]:
                    logging.debug(f"{cplex} esm2 embedding can't match!")
                    return self.__getitem__((idx + 1) % len(self))

            graph1.ndata['f'] = torch.cat([graph1.ndata["f"], graph1_emb], dim=-1)
            graph2.ndata['f'] = torch.cat([graph2.ndata["f"], graph2_emb], dim=-1)

        if "nuv" not in graph1.ndata:
            x1 = graph1.ndata["x"].contiguous()
            x2 = graph2.ndata["x"].contiguous()

            # rotation = Rotation.random(num=1)
            # rotation_matrix = rotation.as_matrix().squeeze()
            # x2 = x2 @ torch.from_numpy(rotation_matrix).float()
            # graph2.ndata["x"] = x2

            batch1 = torch.zeros(x1.shape[0], dtype=torch.int).contiguous()
            batch2 = torch.zeros(x2.shape[0], dtype=torch.int).contiguous()

            normal1, _ = residue_normals(x1, batch1)
            normal2, _ = residue_normals(x2, batch2)
            uv1 = tangent_vectors(normal1)
            uv2 = tangent_vectors(normal2)
            nuv1 = torch.cat([normal1[:, None, :], uv1], dim=-2)
            nuv2 = torch.cat([normal2[:, None, :], uv2], dim=-2)

            graph1.ndata["nuv"] = nuv1.contiguous()
            graph2.ndata["nuv"] = nuv2.contiguous()

            with open(processed_filepath, 'wb') as f:
                pickle.dump(processed_complex, f)

        if self.nuv_residue:
            x1 = graph1.ndata["x"].contiguous()
            x2 = graph2.ndata["x"].contiguous()

            rotation = Rotation.random(num=1)
            rotation_matrix = rotation.as_matrix().squeeze()
            x2 = x2 @ torch.from_numpy(rotation_matrix).float()

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

            graph1.ndata['f'] = torch.cat([graph1.ndata["f"], graph1.ndata["nuv"].flatten(1)], dim=-1)
            graph2.ndata['f'] = torch.cat([graph2.ndata["f"], graph2.ndata["nuv"].flatten(1)], dim=-1)

        if self.nuv_angle:
            normal1 = graph1.ndata["nuv"][:, 0, :]
            normal2 = graph2.ndata["nuv"][:, 0, :]

            graph1.edata['f'] = torch.cat([
                graph1.edata["f"][:, :-1],
                (normal1[graph1.edges()[0]] * normal1[graph1.edges()[1]]).sum(-1, keepdim=True)
            ], dim=-1)
            graph2.edata['f'] = torch.cat([
                graph2.edata["f"][:, :-1],
                (normal2[graph2.edges()[0]] * normal2[graph2.edges()[1]]).sum(-1, keepdim=True)
            ], dim=-1)

        # Optionally zero-out input data for an input-independent pipeline (per Karpathy's suggestion)
        if self.input_indep:
            processed_complex = zero_out_complex_features(processed_complex)
        # Manually filter for desired node and edge features
        # n_feat_idx_1, n_feat_idx_2 = 43, 85  # HSAAC
        # processed_complex['graph1'].ndata['f'] = processed_complex['graph1'].ndata['f'][:, n_feat_idx_1: n_feat_idx_2]
        # processed_complex['graph2'].ndata['f'] = processed_complex['graph2'].ndata['f'][:, n_feat_idx_1: n_feat_idx_2]

        # g1_rsa = processed_complex['graph1'].ndata['f'][:, 35: 36].reshape(-1, 1)  # RSA
        # g1_psaia = processed_complex['graph1'].ndata['f'][:, 37: 43]  # PSAIA
        # g1_hsaac = processed_complex['graph1'].ndata['f'][:, 43: 85]  # HSAAC
        # processed_complex['graph1'].ndata['f'] = torch.cat((g1_rsa, g1_psaia, g1_hsaac), dim=1)
        #
        # g2_rsa = processed_complex['graph2'].ndata['f'][:, 35: 36].reshape(-1, 1)  # RSA
        # g2_psaia = processed_complex['graph2'].ndata['f'][:, 37: 43]  # PSAIA
        # g2_hsaac = processed_complex['graph2'].ndata['f'][:, 43: 85]  # HSAAC
        # processed_complex['graph2'].ndata['f'] = torch.cat((g2_rsa, g2_psaia, g2_hsaac), dim=1)

        # processed_complex['graph1'].edata['f'] = processed_complex['graph1'].edata['f'][:, 1].reshape(-1, 1)
        # processed_complex['graph2'].edata['f'] = processed_complex['graph2'].edata['f'][:, 1].reshape(-1, 1)

        # Return requested complex to DataLoader

        # if processed_complex['graph1'].num_nodes() < processed_complex['graph2'].num_nodes():
        #     data = processed_complex['graph1']
        #     processed_complex['graph1'] = processed_complex['graph2']
        #     processed_complex['graph2'] = data
        #     processed_complex['examples'] = processed_complex['examples'][:, [1, 0, 2]]
        return processed_complex

    def __len__(self) -> int:
        r"""Number of data batches in the dataset."""
        return len(self.filenames_frame)

    @property
    def num_chains(self) -> int:
        """Number of protein chains in each complex."""
        return 2

    @property
    def num_classes(self) -> int:
        """Number of possible classes for each inter-chain residue pair in each complex."""
        return 2

    @property
    def num_node_features(self) -> int:
        """Number of node feature values after encoding them."""
        num_node_features = 113
        if self.nuv_residue:
            num_node_features += 9
        if self.protrans:
            num_node_features += 1024
        if self.esm2:
            num_node_features += 2560
        return num_node_features

    @property
    def num_edge_features(self) -> int:
        """Number of edge feature values after encoding them."""
        return 28

    @property
    def raw_path(self) -> str:
        """Directory in which to locate raw pairs."""
        return self.raw_dir

    @property
    def url(self) -> str:
        """URL with which to download TAR archive of preprocessed pairs (Need to manually download Part B)."""
        return 'https://zenodo.org/record/6299835/files/final_processed_dips.tar.gz.partaa'


class CASPCAPRIDGLDataset(DGLDataset):
    r"""Bound protein complex dataset for DGL with PyTorch.

    Statistics:

    - Test homodimers: 14
    - Test heterodimers: 5
    - Number of structures per complex: 2
    ----------------------
    - Total dimers: 19
    ----------------------

    Parameters
    ----------
    mode: str, optional
        Should be one of ['test']. Default: 'test'.
    raw_dir: str
        Raw file directory to download/contains the input data directory. Default: 'final/raw'.
    knn: int
        How many nearest neighbors to which to connect a given node. Default: 20.
    geo_nbrhd_size: int
        Size of each edge's neighborhood when updating geometric edge features. Default: 2.
    self_loops: bool
        Whether to connect a given node to itself. Default: True.
    percent_to_use: float
        How much of the dataset to load. Default: 1.00.
    process_complexes: bool
        Whether to process each unprocessed complex as we load in the dataset. Default: True.
    input_indep: bool
        Whether to zero-out each input node and edge feature for an input-independent baseline. Default: False.
    force_reload: bool
        Whether to reload the dataset. Default: False.
    verbose: bool
        Whether to print out progress information. Default: False.

    Notes
    -----
    All the samples will be loaded and preprocessed in the memory first.

    Examples
    --------
    >>> # Get dataset
    >>> test_data = CASPCAPRIDGLDataset(mode='test')
    >>>
    >>> len(test_data)
    19
    >>> test_data.num_chains
    2
    """

    def __init__(self,
                 mode='test',
                 raw_dir="../CASP_CAPRI/final/raw",
                 knn=20,
                 geo_nbrhd_size=2,
                 self_loops=True,
                 percent_to_use=1.00,
                 process_complexes=True,
                 input_indep=False,
                 force_reload=False,
                 verbose=False,
                 nuv_residue=True,
                 nuv_angle=False,
                 generate_fasta=False,
                 esm2=False, ):
        assert mode in ['test']
        assert 0.0 < percent_to_use <= 1.0
        self.mode = mode
        self.root = raw_dir
        self.knn = knn
        self.geo_nbrhd_size = geo_nbrhd_size
        self.self_loops = self_loops
        self.percent_to_use = percent_to_use  # How much of the dataset (e.g. CASP-CAPRI training dataset) to use
        self.process_complexes = process_complexes  # Whether to process any unprocessed complexes before training
        self.input_indep = input_indep  # Whether to use an input-independent pipeline to train the model
        self.final_dir = f"{os.sep}".join(self.root.split(os.sep)[:-1])
        self.processed_dir = os.path.join(self.final_dir, 'processed')
        self.nuv_residue = nuv_residue
        self.nuv_angle = nuv_angle
        self.generate_fasta = generate_fasta
        self.ems2 = esm2

        self.filename_sampling = 0.0 < self.percent_to_use < 1.0
        self.base_txt_filename, self.filenames_frame_txt_filename, self.filenames_frame_txt_filepath = \
            construct_filenames_frame_txt_filenames(self.mode, self.percent_to_use, self.filename_sampling, self.root)

        # Try to load the text file containing all CASP-CAPRI filenames, and alert the user if it is missing
        filenames_frame_to_be_written = not os.path.exists(self.filenames_frame_txt_filepath)

        # Randomly sample DataFrame of filenames with requested cross validation ratio
        if self.filename_sampling:
            if filenames_frame_to_be_written:
                try:
                    self.filenames_frame = pd.read_csv(
                        os.path.join(self.root, self.base_txt_filename + '.txt'), header=None)
                except Exception:
                    raise FileNotFoundError(
                        build_filenames_frame_error_message('CASP-CAPRI', 'load', self.filenames_frame_txt_filepath))
                self.filenames_frame = self.filenames_frame.sample(frac=self.percent_to_use).reset_index()
                try:
                    self.filenames_frame[0].to_csv(self.filenames_frame_txt_filepath, header=None, index=None)
                except Exception:
                    raise Exception(
                        build_filenames_frame_error_message('CASP-CAPRI', 'write', self.filenames_frame_txt_filepath))

        # Load in existing DataFrame of filenames as requested (or if a sampled DataFrame .txt has already been written)
        if not filenames_frame_to_be_written:
            try:
                self.filenames_frame = pd.read_csv(self.filenames_frame_txt_filepath, header=None)
            except Exception:
                raise FileNotFoundError(
                    build_filenames_frame_error_message('CASP-CAPRI', 'load', self.filenames_frame_txt_filepath))

        super(CASPCAPRIDGLDataset, self).__init__(name='CASP-CAPRI-Plus',
                                                  raw_dir=raw_dir,
                                                  force_reload=force_reload,
                                                  verbose=verbose)
        logging.info(f"Loaded CASP-CAPRI-Plus {mode}-set, source: {self.processed_dir}, length: {len(self)}")

    def download(self):
        """Download and extract a pre-packaged version of the raw pairs if 'self.raw_dir' is not already populated."""
        # Path to store the file
        gz_file_path = os.path.join(os.path.join(*self.raw_dir.split(os.sep)[:-1]), 'final_raw_casp_capri.tar.gz')

        # Download file
        download(self.url, path=gz_file_path)

        # Check SHA-1
        if not check_sha1(gz_file_path, self._sha1_str):
            raise UserWarning('File {} is downloaded but the content hash does not match.'
                              'The repo may be outdated or download may be incomplete. '
                              'Otherwise you can create an issue for it.'.format(gz_file_path))

        # Remove existing raw directory to make way for the new archive to be extracted
        if os.path.exists(self.raw_dir):
            os.removedirs(self.raw_dir)

        # Extract archive to parent directory of `self.raw_dir`
        self._extract_gz(gz_file_path, os.path.join(*self.raw_dir.split(os.sep)[:-1]))

    def process(self):
        """Process each protein complex into a testing-ready dictionary representing both structures."""
        if self.process_complexes:
            # Ensure the directory of processed complexes is already created
            os.makedirs(self.processed_dir, exist_ok=True)
            # Process each unprocessed protein complex
            for (i, raw_path) in self.filenames_frame.iterrows():
                raw_filepath = os.path.join(self.raw_dir, f'{os.path.splitext(raw_path[0])[0]}.dill')
                processed_filepath = os.path.join(self.processed_dir, f'{os.path.splitext(raw_path[0])[0]}.dill')
                if not os.path.exists(processed_filepath):
                    processed_parent_dir_to_make = os.path.join(self.processed_dir, os.path.split(raw_path[0])[0])
                    os.makedirs(processed_parent_dir_to_make, exist_ok=True)
                    process_complex_into_dict(raw_filepath, processed_filepath, self.knn,
                                              self.geo_nbrhd_size, self.self_loops, check_sequence=False)

    def has_cache(self):
        """Check if each complex is downloaded and available for testing."""
        for (i, raw_path) in self.filenames_frame.iterrows():
            processed_filepath = os.path.join(self.processed_dir, f'{os.path.splitext(raw_path[0])[0]}.dill')
            if not os.path.exists(processed_filepath):
                logging.info(
                    f'Unable to load at least one processed CASP-CAPRI pair. '
                    f'Please make sure all processed pairs have been successfully downloaded and are not corrupted.')
                raise FileNotFoundError
        logging.info('CASP-CAPRI cache found')  # Otherwise, a cache was found!

    def __getitem__(self, idx):
        r""" Get feature dictionary by index of complex.

        Parameters
        ----------
        idx: int

        Returns
        -------
        :class:`dict`

    - ``complex['graph1']:`` DGLGraph (of length M) containing each of the first data's encoded node and edge features
    - ``complex['graph2']:`` DGLGraph (of length N) containing each of the second data's encoded node and edge features
    - ``complex['examples']:`` PyTorch Tensor (of shape (M x N) x 3) containing the labels for inter-data node pairs
    - ``complex['complex']:`` Python string describing the complex's code and original pdb filename
    - ``complex['filepath']:`` Python string describing the complex's filepath
        """
        # Assemble filepath of processed protein complex
        complex_filepath = f'{os.path.splitext(self.filenames_frame[0][idx])[0]}.dill'
        processed_filepath = os.path.join(self.processed_dir, complex_filepath)

        if self.generate_fasta:
            raw_path = self.filenames_frame[0][idx]
            pre_file = "../CASP_CAPRI/interim/pairs-pruned/{}".format(
                raw_path)
            if os.path.exists(pre_file):
                get_sequence(pre_file, "CASP")
            else:
                print(raw_path)

        # Load in processed complex
        with open(processed_filepath, 'rb') as f:
            processed_complex = pickle.load(f)

        # Optionally zero-out input data for an input-independent pipeline (per Karpathy's suggestion)
        if self.input_indep:
            processed_complex = zero_out_complex_features(processed_complex)
        processed_complex['filepath'] = complex_filepath  # Add filepath to each complex dictionary

        graph1 = processed_complex["graph1"]
        graph2 = processed_complex["graph2"]

        if "nuv" not in graph1.ndata:
            x1 = graph1.ndata["x"].contiguous()
            x2 = graph2.ndata["x"].contiguous()

            # rotation = Rotation.random(num=1)
            # rotation_matrix = rotation.as_matrix().squeeze()
            # x2 = x2 @ torch.from_numpy(rotation_matrix).float()
            # graph2.ndata["x"] = x2

            batch1 = torch.zeros(x1.shape[0], dtype=torch.int).contiguous()
            batch2 = torch.zeros(x2.shape[0], dtype=torch.int).contiguous()

            normal1, _ = residue_normals(x1, batch1)
            normal2, _ = residue_normals(x2, batch2)
            uv1 = tangent_vectors(normal1)
            uv2 = tangent_vectors(normal2)
            nuv1 = torch.cat([normal1[:, None, :], uv1], dim=-2)
            nuv2 = torch.cat([normal2[:, None, :], uv2], dim=-2)

            graph1.ndata["nuv"] = nuv1.contiguous()
            graph2.ndata["nuv"] = nuv2.contiguous()

            with open(processed_filepath, 'wb') as f:
                pickle.dump(processed_complex, f)

        if self.nuv_residue:
            graph1.ndata['f'] = torch.cat([graph1.ndata["f"], graph1.ndata["nuv"].flatten(1)], dim=-1)
            graph2.ndata['f'] = torch.cat([graph2.ndata["f"], graph2.ndata["nuv"].flatten(1)], dim=-1)

        if self.nuv_angle:
            normal1 = graph1.ndata["nuv"][:, 0, :]
            normal2 = graph2.ndata["nuv"][:, 0, :]
            graph1.edata['f'] = torch.cat([
                graph1.edata["f"][:, :-1],
                (normal1[graph1.edges()[0]] * normal1[graph1.edges()[1]]).sum(-1, keepdim=True)
            ], dim=-1)
            graph2.edata['f'] = torch.cat([
                graph2.edata["f"][:, :-1],
                (normal2[graph2.edges()[0]] * normal2[graph2.edges()[1]]).sum(-1, keepdim=True)
            ], dim=-1)
        # Manually filter for desired node and edge features
        # n_feat_idx_1, n_feat_idx_2 = 43, 85  # HSAAC
        # processed_complex['graph1'].ndata['f'] = processed_complex['graph1'].ndata['f'][:, n_feat_idx_1: n_feat_idx_2]
        # processed_complex['graph2'].ndata['f'] = processed_complex['graph2'].ndata['f'][:, n_feat_idx_1: n_feat_idx_2]

        # g1_rsa = processed_complex['graph1'].ndata['f'][:, 35: 36].reshape(-1, 1)  # RSA
        # g1_psaia = processed_complex['graph1'].ndata['f'][:, 37: 43]  # PSAIA
        # g1_hsaac = processed_complex['graph1'].ndata['f'][:, 43: 85]  # HSAAC
        # processed_complex['graph1'].ndata['f'] = torch.cat((g1_rsa, g1_psaia, g1_hsaac), dim=1)
        #
        # g2_rsa = processed_complex['graph2'].ndata['f'][:, 35: 36].reshape(-1, 1)  # RSA
        # g2_psaia = processed_complex['graph2'].ndata['f'][:, 37: 43]  # PSAIA
        # g2_hsaac = processed_complex['graph2'].ndata['f'][:, 43: 85]  # HSAAC
        # processed_complex['graph2'].ndata['f'] = torch.cat((g2_rsa, g2_psaia, g2_hsaac), dim=1)

        # processed_complex['graph1'].edata['f'] = processed_complex['graph1'].edata['f'][:, 1].reshape(-1, 1)
        # processed_complex['graph2'].edata['f'] = processed_complex['graph2'].edata['f'][:, 1].reshape(-1, 1)

        # Return requested complex to DataLoader
        # if processed_complex['graph1'].num_nodes() < processed_complex['graph2'].num_nodes():
        #     data = processed_complex['graph1']
        #     processed_complex['graph1'] = processed_complex['graph2']
        #     processed_complex['graph2'] = data
        #     processed_complex['examples'] = processed_complex['examples'][:, [1, 0, 2]]
        return processed_complex

    def __len__(self) -> int:
        r"""Number of data batches in the dataset."""
        return len(self.filenames_frame)

    @property
    def num_chains(self) -> int:
        """Number of protein chains in each complex."""
        return 2

    @property
    def num_classes(self) -> int:
        """Number of possible classes for each inter-chain residue pair in each complex."""
        return 2

    @property
    def num_node_features(self) -> int:
        """Number of node feature values after encoding them."""
        return 113 if not self.nuv_residue else 113 + 9

    @property
    def num_edge_features(self) -> int:
        """Number of edge feature values after encoding them."""
        return 28

    @property
    def raw_path(self) -> str:
        """Directory in which to locate raw pairs."""
        return self.raw_dir

    @property
    def url(self) -> str:
        """URL with which to download TAR archive of preprocessed pairs."""
        return 'https://zenodo.org/record/6299835/files/final_processed_casp_capri.tar.gz'


class DB5DGLDataset(DGLDataset):
    r"""Unbound protein complex dataset for DGL with PyTorch.

    Statistics:

    - Train dimers: 140
    - Validation dimers: 35
    - Test dimers: 55
    - Number of structures per complex: 2
    ----------------------
    - Total dimers: 230
    ----------------------

    Parameters
    ----------
    mode: str, optional
        Should be one of ['train', 'val', 'test']. Default: 'test'.
    raw_dir: str
        Raw file directory to download/contains the input data directory. Default: 'final/raw'.
    knn: int
        How many nearest neighbors to which to connect a given node. Default: 20.
    geo_nbrhd_size: int
        Size of each edge's neighborhood when updating geometric edge features. Default: 2.
    self_loops: bool
        Whether to connect a given node to itself. Default: True.
    percent_to_use: float
        How much of the dataset to load. Default: 1.00.
    process_complexes: bool
        Whether to process each unprocessed complex as we load in the dataset. Default: True.
    input_indep: bool
        Whether to zero-out each input node and edge feature for an input-independent baseline. Default: False.
    force_reload: bool
        Whether to reload the dataset. Default: False.
    verbose: bool
        Whether to print out progress information. Default: False.

    Notes
    -----
    All the samples will be loaded and preprocessed in the memory first.

    Examples
    --------
    >>> # Get dataset
    >>> train_data = DB5DGLDataset(mode='train')
    >>> val_data = DB5DGLDataset(mode='val')
    >>> test_data = DB5DGLDataset()
    >>>
    >>> len(test_data)
    55
    >>> test_data.num_chains
    2
    """

    def __init__(self,
                 mode='test',
                 raw_dir="../DB5/final/raw",
                 knn=20,
                 geo_nbrhd_size=2,
                 self_loops=True,
                 percent_to_use=1.00,
                 process_complexes=True,
                 input_indep=False,
                 force_reload=False,
                 verbose=False,
                 nuv_residue=True,
                 nuv_angle=False,
                 protrans=False,
                 esm2=False,
                 generate_fasta=False, ):
        assert mode in ['train', 'val', 'test']
        assert 0.0 < percent_to_use <= 1.0
        self.mode = mode
        self.root = raw_dir
        self.knn = knn
        self.geo_nbrhd_size = geo_nbrhd_size
        self.self_loops = self_loops
        self.percent_to_use = percent_to_use  # How much of the dataset (e.g. DB5 test dataset) to use
        self.process_complexes = process_complexes  # Whether to process any unprocessed complexes before training
        self.input_indep = input_indep  # Whether to use an input-independent pipeline to train the model
        self.final_dir = f"{os.sep}".join(self.root.split(os.sep)[:-1])
        self.processed_dir = os.path.join(self.final_dir, 'processed')
        self.nuv_residue = nuv_residue
        self.nuv_angle = nuv_angle
        self.protrans = protrans
        self.esm2 = esm2
        self.generate_fasta = generate_fasta

        self.filename_sampling = 0.0 < self.percent_to_use < 1.0
        self.base_txt_filename, self.filenames_frame_txt_filename, self.filenames_frame_txt_filepath = \
            construct_filenames_frame_txt_filenames(self.mode, self.percent_to_use, self.filename_sampling, self.root)

        # Try to load the text file containing all DB5 filenames, and alert the user if it is missing
        filenames_frame_to_be_written = not os.path.exists(self.filenames_frame_txt_filepath)

        # Randomly sample DataFrame of filenames with requested cross validation ratio
        if self.filename_sampling:
            if filenames_frame_to_be_written:
                try:
                    self.filenames_frame = pd.read_csv(
                        os.path.join(self.root, self.base_txt_filename + '.txt'), header=None)
                except Exception:
                    raise FileNotFoundError(
                        build_filenames_frame_error_message('DB5', 'load', self.filenames_frame_txt_filepath))
                self.filenames_frame = self.filenames_frame.sample(frac=self.percent_to_use).reset_index()
                try:
                    self.filenames_frame[0].to_csv(self.filenames_frame_txt_filepath, header=None, index=None)
                except Exception:
                    raise Exception(
                        build_filenames_frame_error_message('DB5', 'write', self.filenames_frame_txt_filepath))

        # Load in existing DataFrame of filenames as requested (or if a sampled DataFrame .txt has already been written)
        if not filenames_frame_to_be_written:
            try:
                self.filenames_frame = pd.read_csv(self.filenames_frame_txt_filepath, header=None)
            except Exception:
                raise FileNotFoundError(
                    build_filenames_frame_error_message('DB5', 'load', self.filenames_frame_txt_filepath))

        super(DB5DGLDataset, self).__init__(name='DB5-Plus',
                                            raw_dir=raw_dir,
                                            force_reload=force_reload,
                                            verbose=verbose)
        print(f"Loaded DB5-Plus {mode}-set, source: {self.processed_dir}, length: {len(self)}")
        if self.protrans:
            self.protrans_emb = {}
            for root, _, files in os.walk("DeepInteract/data/DB5/embedding_protrans"):
                for f in files:
                    cplex, pair, chain_1, chain = f[:-4].split("-")
                    # assert chain_1 == "1"
                    self.protrans_emb.setdefault(cplex, {})
                    self.protrans_emb[cplex].setdefault(int(pair), {})
                    self.protrans_emb[cplex][int(pair)][chain] = os.path.join(root, f)
            drop_index = []
            for (i, raw_path) in self.filenames_frame.iterrows():
                cplex = raw_path[0][3:-5]
                if cplex not in self.protrans_emb:
                    drop_index.append(i)
            logging.info("Loaded prottrans embedding and missing {}".format(len(drop_index)))
            self.filenames_frame = self.filenames_frame.drop(drop_index)

        if self.esm2:
            self.esm2_emb = {}
            for root, _, files in os.walk("DeepInteract/data/DIPS/embedding_esm2_2560"):
                for f in files:
                    cplex, pair, chain_1, chain = f[:-4].split("-")
                    # assert chain_1 == "1"
                    self.esm2_emb.setdefault(cplex, {})
                    self.esm2_emb[cplex].setdefault(int(pair), {})
                    self.esm2_emb[cplex][int(pair)][chain] = os.path.join(root, f)
            drop_index = []
            for (i, raw_path) in self.filenames_frame.iterrows():
                cplex = raw_path[0][3:-5]
                if cplex not in self.esm2_emb:
                    drop_index.append(i)
            logging.info("Loaded esm2 embedding and missing {}".format(len(drop_index)))
            self.filenames_frame = self.filenames_frame.drop(drop_index)

    def download(self):
        """Download and extract a pre-packaged version of the raw pairs if 'self.raw_dir' is not already populated."""
        # Path to store the file
        gz_file_path = os.path.join(os.path.join(*self.raw_dir.split(os.sep)[:-1]), 'final_raw_db5.tar.gz')

        # Download file
        download(self.url, path=gz_file_path)

        # Check SHA-1
        if not check_sha1(gz_file_path, self._sha1_str):
            raise UserWarning('File {} is downloaded but the content hash does not match.'
                              'The repo may be outdated or download may be incomplete. '
                              'Otherwise you can create an issue for it.'.format(gz_file_path))

        # Remove existing raw directory to make way for the new archive to be extracted
        if os.path.exists(self.raw_dir):
            os.removedirs(self.raw_dir)

        # Extract archive to parent directory of `self.raw_dir`
        self._extract_gz(gz_file_path, os.path.join(*self.raw_dir.split(os.sep)[:-1]))

    def process(self):
        """Process each protein complex into a testing-ready dictionary representing both structures."""
        if self.process_complexes:
            # Ensure the directory of processed complexes is already created
            os.makedirs(self.processed_dir, exist_ok=True)
            # Process each unprocessed protein complex
            for (i, raw_path) in self.filenames_frame.iterrows():
                raw_filepath = os.path.join(self.raw_dir, f'{os.path.splitext(raw_path[0])[0]}.dill')
                processed_filepath = os.path.join(self.processed_dir, f'{os.path.splitext(raw_path[0])[0]}.dill')
                if not os.path.exists(processed_filepath):
                    processed_parent_dir_to_make = os.path.join(self.processed_dir, os.path.split(raw_path[0])[0])
                    os.makedirs(processed_parent_dir_to_make, exist_ok=True)
                    process_complex_into_dict(raw_filepath, processed_filepath, self.knn,
                                              self.geo_nbrhd_size, self.self_loops, check_sequence=False)

    def has_cache(self):
        """Check if each complex is downloaded and available for training, validation, or testing."""
        for (i, raw_path) in self.filenames_frame.iterrows():
            processed_filepath = os.path.join(self.processed_dir, f'{os.path.splitext(raw_path[0])[0]}.dill')
            if not os.path.exists(processed_filepath):
                logging.info(
                    f'Unable to load at least one processed DB5 pair. '
                    f'Please make sure all processed pairs have been successfully downloaded and are not corrupted.')
                raise FileNotFoundError
        logging.info('DB5 cache found')  # Otherwise, a cache was found!

    def __getitem__(self, idx):
        r""" Get feature dictionary by index of complex.

        Parameters
        ----------
        idx : int

        Returns
        -------
        :class:`dict`

    - ``complex['graph1']:`` DGLGraph (of length M) containing each of the first data's encoded node and edge features
    - ``complex['graph2']:`` DGLGraph (of length N) containing each of the second data's encoded node and edge features
    - ``complex['examples']:`` PyTorch Tensor (of shape (M x N) x 3) containing the labels for inter-data node pairs
    - ``complex['complex']:`` Python string describing the complex's code and original pdb filename
    - ``complex['filepath']:`` Python string describing the complex's filepath
        """
        # Assemble filepath of processed protein complex
        complex_filepath = f'{os.path.splitext(self.filenames_frame[0][idx])[0]}.dill'
        processed_filepath = os.path.join(self.processed_dir, complex_filepath)

        if self.generate_fasta:
            raw_path = self.filenames_frame[0][idx]
            pre_file = "../DB5/interim/pairs/{}".format(
                raw_path)
            if os.path.exists(pre_file):
                get_sequence(pre_file, "DB5")
            else:
                print(raw_path)

        # Load in processed complex
        with open(processed_filepath, 'rb') as f:
            processed_complex = pickle.load(f)
        processed_complex['filepath'] = complex_filepath  # Add filepath to each complex dictionary

        graph1 = processed_complex["graph1"]
        graph2 = processed_complex["graph2"]

        if "nuv" not in graph1.ndata:
            x1 = graph1.ndata["x"].contiguous()
            x2 = graph2.ndata["x"].contiguous()

            # rotation = Rotation.random(num=1)
            # rotation_matrix = rotation.as_matrix().squeeze()
            # x2 = x2 @ torch.from_numpy(rotation_matrix).float()
            # graph2.ndata["x"] = x2

            batch1 = torch.zeros(x1.shape[0], dtype=torch.int).contiguous()
            batch2 = torch.zeros(x2.shape[0], dtype=torch.int).contiguous()

            normal1, _ = residue_normals(x1, batch1)
            normal2, _ = residue_normals(x2, batch2)
            uv1 = tangent_vectors(normal1)
            uv2 = tangent_vectors(normal2)
            nuv1 = torch.cat([normal1[:, None, :], uv1], dim=-2)
            nuv2 = torch.cat([normal2[:, None, :], uv2], dim=-2)

            graph1.ndata["nuv"] = nuv1.contiguous()
            graph2.ndata["nuv"] = nuv2.contiguous()

            with open(processed_filepath, 'wb') as f:
                pickle.dump(processed_complex, f)

        if self.nuv_residue:
            graph1.ndata['f'] = torch.cat([graph1.ndata["f"], graph1.ndata["nuv"].flatten(1)], dim=-1)
            graph2.ndata['f'] = torch.cat([graph2.ndata["f"], graph2.ndata["nuv"].flatten(1)], dim=-1)

        if self.nuv_angle:
            normal1 = graph1.ndata["nuv"][:, 0, :]
            normal2 = graph2.ndata["nuv"][:, 0, :]
            graph1.edata['f'] = torch.cat([
                graph1.edata["f"][:, :-1],
                (normal1[graph1.edges()[0]] * normal1[graph1.edges()[1]]).sum(-1, keepdim=True)
            ], dim=-1)
            graph2.edata['f'] = torch.cat([
                graph2.edata["f"][:, :-1],
                (normal2[graph2.edges()[0]] * normal2[graph2.edges()[1]]).sum(-1, keepdim=True)
            ], dim=-1)

        for n in graph1.ndata:
            graph1.ndata[n] = graph1.ndata[n].contiguous()
        for n in graph1.edata:
            graph1.edata[n] = graph1.edata[n].contiguous()
        for n in graph2.ndata:
            graph2.ndata[n] = graph2.ndata[n].contiguous()
        for n in graph2.edata:
            graph2.edata[n] = graph2.edata[n].contiguous()
        if self.mode == "train" and (
                graph1.num_nodes() > 1024 or graph2.num_nodes() > 1024 or graph1.num_nodes() * graph2.num_nodes() > 65536):
            return self[torch.randint(0, len(self), (1,)).item()]
        # if graph1.num_nodes() < graph2.num_nodes():
        #     graph = processed_complex['graph1']
        #     processed_complex['graph1'] = processed_complex['graph2']
        #     processed_complex['graph2'] = graph
        #     examples = processed_complex['examples'][:, [1, 0, 2]]
        #     fisrtsort = (examples[:, 0] * graph.num_nodes() + examples[:, 1]).argsort()
        #     examples = examples[fisrtsort]
        #     processed_complex['examples'] = examples

        # Optionally zero-out input data for an input-independent pipeline (per Karpathy's suggestion)
        # Manually filter for desired node and edge features
        # n_feat_idx_1, n_feat_idx_2 = 43, 85  # HSAAC
        # processed_complex['graph1'].ndata['f'] = processed_complex['graph1'].ndata['f'][:, n_feat_idx_1: n_feat_idx_2]
        # processed_complex['graph2'].ndata['f'] = processed_complex['graph2'].ndata['f'][:, n_feat_idx_1: n_feat_idx_2]

        # g1_rsa = processed_complex['graph1'].ndata['f'][:, 35: 36].reshape(-1, 1)  # RSA
        # g1_psaia = processed_complex['graph1'].ndata['f'][:, 37: 43]  # PSAIA
        # g1_hsaac = processed_complex['graph1'].ndata['f'][:, 43: 85]  # HSAAC
        # processed_complex['graph1'].ndata['f'] = torch.cat((g1_rsa, g1_psaia, g1_hsaac), dim=1)
        #
        # g2_rsa = processed_complex['graph2'].ndata['f'][:, 35: 36].reshape(-1, 1)  # RSA
        # g2_psaia = processed_complex['graph2'].ndata['f'][:, 37: 43]  # PSAIA
        # g2_hsaac = processed_complex['graph2'].ndata['f'][:, 43: 85]  # HSAAC
        # processed_complex['graph2'].ndata['f'] = torch.cat((g2_rsa, g2_psaia, g2_hsaac), dim=1)

        # processed_complex['graph1'].edata['f'] = processed_complex['graph1'].edata['f'][:, 1].reshape(-1, 1)
        # processed_complex['graph2'].edata['f'] = processed_complex['graph2'].edata['f'][:, 1].reshape(-1, 1)

        # Return requested complex to DataLoader
        return processed_complex

    def __len__(self) -> int:
        r"""Number of data batches in the dataset."""
        return len(self.filenames_frame)

    @property
    def num_chains(self) -> int:
        """Number of protein chains in each complex."""
        return 2

    @property
    def num_classes(self) -> int:
        """Number of possible classes for each inter-chain residue pair in each complex."""
        return 2

    @property
    def num_node_features(self) -> int:
        """Number of node feature values after encoding them."""
        return 113 if not self.nuv_residue else 113 + 9

    @property
    def num_edge_features(self) -> int:
        """Number of edge feature values after encoding them."""
        return 28

    @property
    def raw_path(self) -> str:
        """Directory in which to locate raw pairs."""
        return self.raw_dir

    @property
    def url(self) -> str:
        """URL with which to download TAR archive of preprocessed pairs."""
        return 'https://zenodo.org/record/6299835/files/final_processed_db5.tar.gz'


class OtherTestDataset(DGLDataset):
    def __init__(self,
                 mode='test',
                 raw_dir="../datasets/TimeSplits",
                 knn=20,
                 geo_nbrhd_size=2,
                 self_loops=True,
                 percent_to_use=1.00,
                 process_complexes=True,
                 input_indep=False,
                 force_reload=False,
                 verbose=False,
                 nuv_residue=True,
                 nuv_angle=False,
                 generate_fasta=False,
                 esm2=False,
                 paired=False,
                 task=""):
        assert mode in ['train', 'val', 'test', 'all']
        assert 0.0 < percent_to_use <= 1.0
        self.task = task
        pdb_ids = []

        if task == "antibody":
            info = pd.read_csv(os.path.join(raw_dir, "/SAbDab/SAbDab_train_val_count.csv"))
            if mode == "test":
                pdb_ids_file = os.path.join(raw_dir, "SAbDab/SAbDab_test.txt")
                for raw in open(pdb_ids_file, "r").readlines():
                    pdb_ids.append("{}_LHB.pkl".format(raw.strip()))
            else:
                pdb_ids_file = os.path.join(raw_dir, "SAbDab/SAbDab_{}.txt".format(mode))
                for raw in open(pdb_ids_file, "r").readlines():
                    pdb_id = raw.strip()
                    info_raw = info[info["id"] == pdb_id[:-7] + "HL.pdb"]
                    if info_raw["graph1"].item() < 1024 and info_raw["graph2"].item() < 1024 and info_raw[
                        "graph12"].item() < 65536:
                        pdb_ids.append(raw.strip())

        elif task in ["db5.5", "db5.5_all"]:
            raw_dir = "data/DB5.5"
            if mode == 'all':
                for _model in ["train", "val", "test"]:
                    for raw in open(os.path.join(raw_dir, "{}.txt".format(_model)), "r").readlines():
                        pdb_id = "{}_{}_{}.pkl".format(*raw.strip().split(" "))
                        file_dir = os.path.join(raw_dir, "data", pdb_id)
                        if os.path.exists(file_dir):
                            pdb_ids.append(pdb_id)
            else:
                for raw in open(os.path.join(raw_dir, "{}.txt".format(mode)), "r").readlines():
                    pdb_id = "{}_{}_{}.pkl".format(*raw.strip().split(" "))
                    if mode in ["test"]:
                        pdb_ids.append(pdb_id)
                        continue
                    else:
                        file_dir = os.path.join(raw_dir, "data", pdb_id)
                        if os.path.exists(file_dir):
                            with open(file_dir, 'rb') as f:
                                processed_complex = pickle.load(f)
                                graph1 = processed_complex["graph1"]
                                graph2 = processed_complex["graph2"]
                                if graph1.num_nodes() < 1024 and graph2.num_nodes() < 1024 and \
                                        graph1.num_nodes() * graph2.num_nodes() < 65536:
                                    pdb_ids.append(pdb_id)
    
        elif task[:6] == "casp15":
            raw_dir = "data/CASP"
            assert mode == "test"
            if task[6] == "o":
                for raw in open(os.path.join(raw_dir, "{}_Homo.txt".format(mode)), "r").readlines():
                    pdb_id = "{}_{}_{}.pkl".format(*raw.strip().split(" "))
                    pdb_ids.append(pdb_id)
            elif task[6] == "e":
                for raw in open(os.path.join(raw_dir, "{}_Hetero.txt".format(mode)), "r").readlines():
                    pdb_id = "{}_{}_{}.pkl".format(*raw.strip().split(" "))
                    pdb_ids.append(pdb_id)
            else:
                for raw in open(os.path.join(raw_dir, "{}_Hetero.txt".format(mode)), "r").readlines():
                    pdb_id = "{}_{}_{}.pkl".format(*raw.strip().split(" "))
                    pdb_ids.append(pdb_id)
                for raw in open(os.path.join(raw_dir, "{}_Homo.txt".format(mode)), "r").readlines():
                    pdb_id = "{}_{}_{}.pkl".format(*raw.strip().split(" "))
                    pdb_ids.append(pdb_id)
        
        elif task == "timesplits":
            for raw in tqdm(open(os.path.join("data", "{}.txt".format(mode)), "r").readlines(),
                            desc="Loading data {}".format(mode)):
                if mode == "train":
                    pdb_id = "{}.pkl".format(raw.strip().replace(" ", "_")).lower()
                    file_dir = os.path.join(raw_dir, "data", pdb_id)
                    if os.path.exists(file_dir):
                        pdb_ids.append(pdb_id)
                        # with open(file_dir, 'rb') as fin:
                        #     data = pickle.load(fin)
                        # graph1 = data['graph1']
                        # graph2 = data['graph2']
                        # if graph1.num_nodes() * graph2.num_nodes() < 65536:
                        #     pdb_ids.append(pdb_id)
                        # if graph1.num_nodes() < 512 and graph2.num_nodes() < 512:
                        #     pdb_ids.append(pdb_id)

                elif mode == "val":
                    pdb, chain_a, chain_b = raw.strip().split(" ")
                    pdb_id = "{}_{}_{}.pkl".format(pdb.lower(), chain_a, chain_b)
                    file_dir = os.path.join(raw_dir, "test_data", pdb_id)
                    if os.path.exists(file_dir):
                            pdb_ids.append(pdb_id)

                else:
                    pdb, chain_a, chain_b = raw.strip().split(" ")
                    pdb_id = "{}_{}_{}.pkl".format(pdb.lower(), chain_a, chain_b)
                    file_dir = os.path.join(raw_dir, "test_data", pdb_id)
                    if os.path.exists(file_dir):
                            pdb_ids.append(pdb_id)
            logging.info(f"Loaded TimeSplits {mode}-set,  length: {len(pdb_ids)}")
    
        else:
            raise NotImplementedError

        self.mode = mode
        self.root = raw_dir
        self.knn = knn
        self.geo_nbrhd_size = geo_nbrhd_size
        self.self_loops = self_loops
        self.percent_to_use = percent_to_use  # How much of the dataset (e.g. CASP-CAPRI training dataset) to use
        self.process_complexes = process_complexes  # Whether to process any unprocessed complexes before training
        self.input_indep = input_indep  # Whether to use an input-independent pipeline to train the model
        self.final_dir = f"{os.sep}".join(self.root.split(os.sep))
        self.processed_dir = os.path.join(self.final_dir, 'processed')
        self.nuv_residue = nuv_residue
        self.nuv_angle = nuv_angle
        self.generate_fasta = generate_fasta
        self.esm2 = esm2
        self.paired = paired

        self.filenames_frame = []
        for pdb_id in pdb_ids:
            self.filenames_frame.append(os.path.join(raw_dir, "data", pdb_id))


        super(OtherTestDataset, self).__init__(name='AntiBody-Plus',
                                               raw_dir=raw_dir,
                                               force_reload=force_reload,
                                               verbose=verbose)
        

    def download(self):
        """Download and extract a pre-packaged version of the raw pairs if 'self.raw_dir' is not already populated."""
        # Path to store the file
        gz_file_path = os.path.join(os.path.join(*self.raw_dir.split(os.sep)[:-1]), 'final_raw_casp_capri.tar.gz')

        # Download file
        download(self.url, path=gz_file_path)

        # Check SHA-1
        if not check_sha1(gz_file_path, self._sha1_str):
            raise UserWarning('File {} is downloaded but the content hash does not match.'
                              'The repo may be outdated or download may be incomplete. '
                              'Otherwise you can create an issue for it.'.format(gz_file_path))

        # Remove existing raw directory to make way for the new archive to be extracted
        if os.path.exists(self.raw_dir):
            os.removedirs(self.raw_dir)

        # Extract archive to parent directory of `self.raw_dir`
        self._extract_gz(gz_file_path, os.path.join(*self.raw_dir.split(os.sep)[:-1]))

    def process(self):
        """Process each protein complex into a testing-ready dictionary representing both structures."""
        if self.process_complexes:
            # Ensure the directory of processed complexes is already created
            os.makedirs(self.processed_dir, exist_ok=True)
            # Process each unprocessed protein complex
            for (i, raw_path) in self.filenames_frame.iterrows():
                raw_filepath = os.path.join(self.raw_dir, f'{os.path.splitext(raw_path[0])[0]}.dill')
                processed_filepath = os.path.join(self.processed_dir, f'{os.path.splitext(raw_path[0])[0]}.dill')
                if not os.path.exists(processed_filepath):
                    processed_parent_dir_to_make = os.path.join(self.processed_dir, os.path.split(raw_path[0])[0])
                    os.makedirs(processed_parent_dir_to_make, exist_ok=True)
                    process_complex_into_dict(raw_filepath, processed_filepath, self.knn,
                                              self.geo_nbrhd_size, self.self_loops, check_sequence=False)

    def has_cache(self):
        return True

    def __getitem__(self, idx):
        r""" Get feature dictionary by index of complex.

        Parameters
        ----------
        idx: int

        Returns
        -------
        :class:`dict`

    - ``complex['graph1']:`` DGLGraph (of length M) containing each of the first data's encoded node and edge features
    - ``complex['graph2']:`` DGLGraph (of length N) containing each of the second data's encoded node and edge features
    - ``complex['examples']:`` PyTorch Tensor (of shape (M x N) x 3) containing the labels for inter-data node pairs
    - ``complex['complex']:`` Python string describing the complex's code and original pdb filename
    - ``complex['filepath']:`` Python string describing the complex's filepath
        """
        # Assemble filepath of processed protein complex
        processed_filepath = self.filenames_frame[idx]
        # pdb_id = processed_filepath.split("/")[-1].split(".")[0]
        # # Load in processed complex
        # new_path = processed_filepath.split("/")
        # new_path[-1] = new_path[-1].lower()
        # with open("/"+os.path.join(*new_path), 'rb') as f:
        #     processed_complex = pickle.load(f)
        with open(processed_filepath, 'rb') as f:
            processed_complex = pickle.load(f)

        # Optionally zero-out input data for an input-independent pipeline (per Karpathy's suggestion)
        if self.input_indep:
            processed_complex = zero_out_complex_features(processed_complex)

        graph1 = processed_complex["graph1"]
        graph2 = processed_complex["graph2"]

        if "nuv" not in graph1.ndata:
            x1 = graph1.ndata["x"].contiguous()
            x2 = graph2.ndata["x"].contiguous()

            # rotation = Rotation.random(num=1)
            # rotation_matrix = rotation.as_matrix().squeeze()
            # x2 = x2 @ torch.from_numpy(rotation_matrix).float()
            # graph2.ndata["x"] = x2

            batch1 = torch.zeros(x1.shape[0], dtype=torch.int).contiguous()
            batch2 = torch.zeros(x2.shape[0], dtype=torch.int).contiguous()

            normal1, _ = residue_normals(x1, batch1)
            normal2, _ = residue_normals(x2, batch2)
            uv1 = tangent_vectors(normal1)
            uv2 = tangent_vectors(normal2)
            nuv1 = torch.cat([normal1[:, None, :], uv1], dim=-2)
            nuv2 = torch.cat([normal2[:, None, :], uv2], dim=-2)

            graph1.ndata["nuv"] = nuv1.contiguous()
            graph2.ndata["nuv"] = nuv2.contiguous()

            # with open(processed_filepath, 'wb') as f:
            #     pickle.dump(processed_complex, f)

        if self.nuv_residue:
            graph1.ndata['f'] = torch.cat([graph1.ndata["f"], graph1.ndata["nuv"].flatten(1)], dim=-1)
            graph2.ndata['f'] = torch.cat([graph2.ndata["f"], graph2.ndata["nuv"].flatten(1)], dim=-1)

        if self.nuv_angle:
            normal1 = graph1.ndata["nuv"][:, 0, :]
            normal2 = graph2.ndata["nuv"][:, 0, :]
            graph1.edata['f'] = torch.cat([
                graph1.edata["f"][:, :-1],
                (normal1[graph1.edges()[0]] * normal1[graph1.edges()[1]]).sum(-1, keepdim=True)
            ], dim=-1)
            graph2.edata['f'] = torch.cat([
                graph2.edata["f"][:, :-1],
                (normal2[graph2.edges()[0]] * normal2[graph2.edges()[1]]).sum(-1, keepdim=True)
            ], dim=-1)

        # if self.esm2:
        #     graph1_emb = graph1.ndata['esm']
        #     graph2_emb = graph2.ndata['esm']

        #     graph1.ndata['f'] = torch.cat([graph1.ndata["f"], graph1_emb], dim=-1)
        #     graph2.ndata['f'] = torch.cat([graph2.ndata["f"], graph2_emb], dim=-1)

        if self.paired: 
            rec1d = graph1.ndata['esm_msa_1b']
            lig1d = graph2.ndata['esm_msa_1b']

            graph1.ndata['f'] = torch.cat([graph1.ndata["f"], rec1d], dim=-1)
            graph2.ndata['f'] = torch.cat([graph2.ndata["f"], lig1d], dim=-1)

            com2d = graph1.ndata['pair_feats']
            graph1.ndata['pair_feats'] = com2d.reshape(graph1.num_nodes(), graph2.num_nodes(), -1).float()
            # graph2.ndata['pair_feats'] = torch.from_numpy(com2d).reshape(graph2.num_nodes(), graph1.num_nodes(), -1).float()


        # Manually filter for desired node and edge features
        # n_feat_idx_1, n_feat_idx_2 = 43, 85  # HSAAC
        # processed_complex['graph1'].ndata['f'] = processed_complex['graph1'].ndata['f'][:, n_feat_idx_1: n_feat_idx_2]
        # processed_complex['graph2'].ndata['f'] = processed_complex['graph2'].ndata['f'][:, n_feat_idx_1: n_feat_idx_2]

        # g1_rsa = processed_complex['graph1'].ndata['f'][:, 35: 36].reshape(-1, 1)  # RSA
        # g1_psaia = processed_complex['graph1'].ndata['f'][:, 37: 43]  # PSAIA
        # g1_hsaac = processed_complex['graph1'].ndata['f'][:, 43: 85]  # HSAAC
        # processed_complex['graph1'].ndata['f'] = torch.cat((g1_rsa, g1_psaia, g1_hsaac), dim=1)
        #
        # g2_rsa = processed_complex['graph2'].ndata['f'][:, 35: 36].reshape(-1, 1)  # RSA
        # g2_psaia = processed_complex['graph2'].ndata['f'][:, 37: 43]  # PSAIA
        # g2_hsaac = processed_complex['graph2'].ndata['f'][:, 43: 85]  # HSAAC
        # processed_complex['graph2'].ndata['f'] = torch.cat((g2_rsa, g2_psaia, g2_hsaac), dim=1)

        # processed_complex['graph1'].edata['f'] = processed_complex['graph1'].edata['f'][:, 1].reshape(-1, 1)
        # processed_complex['graph2'].edata['f'] = processed_complex['graph2'].edata['f'][:, 1].reshape(-1, 1)

        # Return requested complex to DataLoade
        if self.task not in ["db5.5", "db5.5_all", "antibody", "propair", "casp15a", "casp15o", "casp15e", "timesplits"]:
            label_file = pd.read_csv(
                "data/OtherTest/{}/true_pdb/h_dist/{}.htxt".format(self.task, pdb_id),
                sep=" ", header=None
            ).values
            label_file = label_file[:graph1.num_nodes()][:, graph1.num_nodes():]
            label = []
            rows, cols = graph1.num_nodes(), graph2.num_nodes()
            for i in range(rows):
                for j in range(cols):
                    if label_file[i][j] <= 6:
                        label.append((i, j, 1))
            processed_complex["filepath"] = "data/OtherTest/{}/cdpred_output/average/{}.htxt".format(self.task, pdb_id)
        else:
            label = processed_complex["examples"]
        exam = torch.zeros((graph1.num_nodes(), graph2.num_nodes())).int()
        for raw in label:
            exam[raw[0], raw[1]] = raw[2]
        labels = []
        for i in range(graph1.num_nodes()):
            for j in range(graph2.num_nodes()):
                labels.append((i, j, exam[i, j].item()))
        processed_complex["examples"] = torch.LongTensor(labels)
        # if graph1.num_nodes() < graph2.num_nodes():
        #     graph = processed_complex['graph1']
        #     processed_complex['graph1'] = processed_complex['graph2']
        #     processed_complex['graph2'] = graph
        #     examples = processed_complex['examples'][:, [1, 0, 2]]
        #     fisrtsort = (examples[:, 0] * graph.num_nodes() + examples[:, 1]).argsort()
        #     examples = examples[fisrtsort]
        #     processed_complex['examples'] = examples
        # print(processed_complex['graph1'].num_nodes(), processed_complex['graph2'].num_nodes())
        return processed_complex

    def __len__(self) -> int:
        r"""Number of data batches in the dataset."""
        return len(self.filenames_frame)

    @property
    def num_chains(self) -> int:
        """Number of protein chains in each complex."""
        return 2

    @property
    def num_classes(self) -> int:
        """Number of possible classes for each inter-chain residue pair in each complex."""
        return 2

    @property
    def num_node_features(self) -> int:
        """Number of node feature values after encoding them."""
        num_node_features = 113
        if self.nuv_residue:
            num_node_features += 9
        # if self.protrans:
        #     num_node_features += 1024
        if self.esm2:
            num_node_features += 2560
        return num_node_features

    @property
    def num_edge_features(self) -> int:
        """Number of edge feature values after encoding them."""
        return 28

    @property
    def raw_path(self) -> str:
        """Directory in which to locate raw pairs."""
        return self.raw_dir
