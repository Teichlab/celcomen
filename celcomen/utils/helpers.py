import numpy as np
import torch

# define a function to derive the gex from the sphex
def calc_gex(sphex):
    """
    Converts a spherical gene expression matrix into a standard gene expression matrix.

    The function takes a spherical expression matrix (`sphex`) and calculates the corresponding 
    gene expression matrix (`gex`) using trigonometric functions such as sine and cosine. 
    The last gene is computed using sine, and the others are computed using cosine, followed by 
    multiplying by the sine of all preceding dimensions.

    Parameters
    ----------
    sphex : torch.Tensor
        The spherical gene expression matrix. Expected shape is (n_samples, n_features - 1).

    Returns
    -------
    gex : torch.Tensor
        The computed gene expression matrix of shape (n_samples, n_features), where `n_features` is one more than in `sphex`.
        All NaN values are replaced with 0.

    Examples
    --------
    >>> sphex = torch.tensor([[0.5, 0.6], [0.3, 0.4]])
    >>> gex = calc_gex(sphex)
    >>> print(gex)
    tensor([[0.8776, 0.5646, 0.4794],
            [0.9553, 0.5646, 0.2955]])
    """
    # setup the gex
    n_genes = sphex.shape[1]+1
    gex = torch.from_numpy(np.zeros((sphex.shape[0], n_genes)).astype('float32'))
    # compute the gex
    for idx in range(n_genes):
        if idx == n_genes-1:
            gex[:,idx] = torch.sin(sphex[:,idx-1])
        else:
            gex[:,idx] = torch.cos(sphex[:,idx])
        for idx_ in range(idx):
            gex[:,idx] *= torch.sin(sphex[:,idx_])
    return torch.nan_to_num(gex)

# define a function to gather positions
def get_pos(n_x, n_y):
    """
    Generates a 2D hexagonal grid of positions.

    This function creates a hexagonal lattice for a 2D grid, where the x-coordinates are adjusted
    for alternating rows. The y-coordinates are spaced based on a predefined step size derived 
    from the geometry of a hexagonal grid.

    Parameters
    ----------
    n_x : int
        Number of positions along the x-axis.
    n_y : int
        Number of positions along the y-axis.

    Returns
    -------
    pos : numpy.ndarray
        A 2D array of shape (n_x * n_y, 2) representing the coordinates of the positions on the hexagonal grid.

    Examples
    --------
    >>> pos = get_pos(3, 3)
    >>> print(pos)
    array([[0.5, 0. ],
           [1.5, 0. ],
           [2.5, 0. ],
           [0. , 1.11803399],
           [1. , 1.11803399],
           [2. , 1.11803399],
           [0.5, 2.23606798],
           [1.5, 2.23606798],
           [2.5, 2.23606798]])
    """
    # create the hex lattice
    xs = np.array([np.arange(0, n_x) + 0.5 if idx % 2 == 0 else np.arange(0, n_x) for idx in range(n_y)])
    # derive the y-step given a distance of one
    y_step = np.sqrt(1**2+0.5**2)
    ys = np.array([[y_step * idy] * n_x for idy in range(n_y)])
    # define the positions
    pos = np.vstack([xs.flatten(), ys.flatten()]).T
    return pos


# define a function to normalize the g2g
def normalize_g2g(g2g):
    """
    Symmetrizes and normalizes a gene-to-gene (G2G) interaction matrix.

    This function ensures that the matrix is symmetrical, normalizes values to be between 0 and 1, 
    and forces the diagonal to be 1 (representing self-interactions).

    Parameters
    ----------
    g2g : numpy.ndarray
        The gene-to-gene interaction matrix, typically of shape (n_genes, n_genes).

    Returns
    -------
    g2g : numpy.ndarray
        The normalized and symmetrized gene-to-gene interaction matrix.

    Examples
    --------
    >>> g2g = np.array([[0.8, 0.2], [0.1, 0.7]])
    >>> normalized_g2g = normalize_g2g(g2g)
    >>> print(normalized_g2g)
    array([[1. , 0.15],
           [0.15, 1. ]])
    """
    # symmetrize the values
    g2g = (g2g + g2g.T) / 2
    # force them to be between 0-1
    g2g[g2g < 0] = 0
    g2g[g2g > 1] = 1
    # force the central line to be 1
    for idx in range(len(g2g)):
        g2g[idx, idx] = 1
    return g2g

# define a function to normalize the g2g
def symmetrize_g2g(g2g):
    """
    Symmetrizes and normalizes a gene-to-gene (G2G) interaction matrix.

    This function ensures that the matrix is symmetrical.

    Parameters
    ----------
    g2g : numpy.ndarray
        The gene-to-gene interaction matrix, typically of shape (n_genes, n_genes).

    Returns
    -------
    g2g : numpy.ndarray
        The normalized and symmetrized gene-to-gene interaction matrix.

    Examples
    --------
    >>> g2g = np.array([[0.8, 0.2], [0.1, 0.7]])
    >>> normalized_g2g = normalize_g2g(g2g)
    >>> print(normalized_g2g)
    array([[1. , 0.15],
           [0.15, 1. ]])
    """
    # symmetrize the values
    g2g = (g2g + g2g.T) / 2

    return g2g

# define a function to derive the gex from the sphex
def calc_sphex(gex):
    """
    Converts a standard gene expression matrix into a spherical expression matrix.

    This function calculates the spherical representation of a gene expression matrix (`gex`), 
    where the new features are derived using trigonometric functions such as arcsin and arccos.

    Parameters
    ----------
    gex : torch.Tensor
        The standard gene expression matrix, of shape (n_samples, n_genes).

    Returns
    -------
    sphex : torch.Tensor
        The spherical gene expression matrix, of shape (n_samples, n_genes - 1).

    Examples
    --------
    >>> gex = torch.tensor([[0.8, 0.6, 0.4], [0.7, 0.5, 0.3]])
    >>> sphex = calc_sphex(gex)
    >>> print(sphex)
    tensor([[1.5708, 0.5404],
            [1.4706, 0.5236]])
    """
    # setup the gex
    n_sgenes = gex.shape[1]-1
    sphex = torch.from_numpy(np.zeros((gex.shape[0], n_sgenes)).astype('float32'))
    # compute the gex
    for idx in range(n_sgenes):
        sphex[:,idx] = gex[:,idx]
        for idx_ in range(idx):
            sphex[:,idx] /= torch.sin(sphex[:,idx_])
        sphex[:,idx] = torch.arccos(sphex[:,idx])
    return sphex
