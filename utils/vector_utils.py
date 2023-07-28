import torch
import torch.nn.functional as F
from pykeops.torch import LazyTensor


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from dMaSIF (https://github.com/FreyrS/dMaSIF):
# -------------------------------------------------------------------------------------------------------------------------------------

def ranges_slices(batch):
    """Helper function for the diagonal ranges function."""
    Ns = batch.bincount()
    indices = Ns.cumsum(0)
    ranges = torch.cat((0 * indices[:1], indices))
    ranges = (
        torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous().to(batch.device)
    )
    slices = (1 + torch.arange(len(Ns))).int().to(batch.device)

    return ranges, slices


def diagonal_ranges(batch_x=None, batch_y=None):
    """Encodes the block-diagonal structure associated to a batch vector."""

    if batch_x is None and batch_y is None:
        return None  # No batch processing
    elif batch_y is None:
        batch_y = batch_x  # "symmetric" case

    ranges_x, slices_x = ranges_slices(batch_x)
    ranges_y, slices_y = ranges_slices(batch_y)

    return ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x


def soft_dimension(features):
    """Continuous approximation of the rank of a (N, D) sample.
    Let "s" denote the (D,) vector of eigenvalues of Cov,
    the (D, D) covariance matrix of the sample "features".
    Then,
        R(features) = \sum_i sqrt(s_i) / \max_i sqrt(s_i)
    This quantity encodes the number of PCA components that would be
    required to describe the sample with a good precision.
    It is equal to D if the sample is isotropic, but is generally much lower.
    Up to the re-normalization by the largest eigenvalue,
    this continuous pseudo-rank is equal to the nuclear norm of the sample.
    """

    nfeat = features.shape[-1]
    features = features.view(-1, nfeat)
    x = features - torch.mean(features, dim=0, keepdim=True)
    cov = x.T @ x
    try:
        u, s, v = torch.svd(cov)
        R = s.sqrt().sum() / s.sqrt().max()
    except:
        return -1
    return R.item()


def soft_distances(x, y, batch_x, batch_y, smoothness=1.0, atomtypes=None):
    """Computes a soft distance function to the atom centers of a protein.
    Implements Eq. (1) of the paper in a fast and numerically stable way.
    Args:
        x (Tensor): (N,3) atom centers.
        y (Tensor): (M,3) sampling locations.
        batch_x (integer Tensor): (N,) batch vector for x, as in PyTorch_geometric.
        batch_y (integer Tensor): (M,) batch vector for y, as in PyTorch_geometric.
        smoothness (float, optional): atom radii if atom types are not provided. Defaults to 1.
        atomtypes (integer Tensor, optional): (N,6) one-hot encoding of the atom chemical types. Defaults to None.
    Returns:
        Tensor: (M,) values of the soft distance function on the points `y`.
    """
    # Build the (N, M, 1) symbolic matrix of squared distances:
    x_i = LazyTensor(x[:, None, :])  # (N, 1, 3) atoms
    y_j = LazyTensor(y[None, :, :])  # (1, M, 3) sampling points
    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (N, M, 1) squared distances

    # Use a block-diagonal sparsity mask to support heterogeneous batch processing:
    D_ij.ranges = diagonal_ranges(batch_x, batch_y)

    if atomtypes is not None:
        # Turn the one-hot encoding "atomtypes" into a vector of diameters "smoothness_i":
        # (N, 6)  -> (N, 1, 1)  (There are 6 atom types)
        atomic_radii = torch.cuda.FloatTensor(
            [170, 110, 152, 155, 180, 190], device=x.device
        )
        atomic_radii = atomic_radii / atomic_radii.min()
        atomtype_radii = atomtypes * atomic_radii[None, :]  # n_atoms, n_atomtypes
        # smoothness = atomtypes @ atomic_radii  # (N, 6) @ (6,) = (N,)
        smoothness = torch.sum(
            smoothness * atomtype_radii, dim=1, keepdim=False
        )  # n_atoms, 1
        smoothness_i = LazyTensor(smoothness[:, None, None])

        # Compute an estimation of the mean smoothness in a neighborhood
        # of each sampling point:
        # density = (-D_ij.sqrt()).exp().sum(0).view(-1)  # (M,) local density of atoms
        # smooth = (smoothness_i * (-D_ij.sqrt()).exp()).sum(0).view(-1)  # (M,)
        # mean_smoothness = smooth / density  # (M,)

        # soft_dists = -mean_smoothness * (
        #    (-D_ij.sqrt() / smoothness_i).logsumexp(dim=0)
        # ).view(-1)
        mean_smoothness = (-D_ij.sqrt()).exp().sum(0)
        mean_smoothness_j = LazyTensor(mean_smoothness[None, :, :])
        mean_smoothness = (
            smoothness_i * (-D_ij.sqrt()).exp() / mean_smoothness_j
        )  # n_atoms, n_points, 1
        mean_smoothness = mean_smoothness.sum(0).view(-1)
        soft_dists = -mean_smoothness * (
            (-D_ij.sqrt() / smoothness_i).logsumexp(dim=0)
        ).reshape(-1)

    else:
        soft_dists = -smoothness * ((-D_ij.sqrt() / smoothness).logsumexp(dim=0)).reshape(
            -1
        )

    return soft_dists


def residue_normals(residues, batch, ):
    with torch.enable_grad():
        residues.requires_grad = False
        residues_g = residues[:].detach()
        residues_g.requires_grad = True
        dists = soft_distances(
                residues,
                residues_g,
                batch,
                batch,
        )
        Loss = (1.0 * dists).sum()
        g = torch.autograd.grad(Loss, residues_g)[0]
        normals = F.normalize(g, p=2, dim=-1)  # (N, 3)
    return normals, batch


def tangent_vectors(normals):
    """Returns a pair of vector fields u and v to complete the orthonormal basis [n,u,v].
          normals        ->             uv
    (N, 3) or (N, S, 3)  ->  (N, 2, 3) or (N, S, 2, 3)
    This routine assumes that the 3D "normal" vectors are normalized.
    It is based on the 2017 paper from Pixar, "Building an orthonormal basis, revisited".
    Args:
        normals (Tensor): (N,3) or (N,S,3) normals `n_i`, i.e. unit-norm 3D vectors.
    Returns:
        (Tensor): (N,2,3) or (N,S,2,3) unit vectors `u_i` and `v_i` to complete
            the tangent coordinate systems `[n_i,u_i,v_i].
    """
    x, y, z = normals[..., 0], normals[..., 1], normals[..., 2]
    s = (2 * (z >= 0)) - 1.0  # = z.sign(), but =1. if z=0.
    a = -1 / (s + z)
    b = x * y * a
    uv = torch.stack((1 + s * x * x * a, s * b, -s * x, b, s + y * y * a, -y), dim=-1)
    uv = uv.view(uv.shape[:-1] + (2, 3))

    return uv
