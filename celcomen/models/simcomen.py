from torch_geometric.nn import GCNConv
from sklearn.neighbors import kneighbors_graph
import torch
import numpy as np
from typing import Optional
#from ..utils.helpers import calc_gex

# define the number of neighbors (six for visium)
n_neighbors = 6
# define the simcomen class
class simcomen(torch.nn.Module):
    """
    A k-hop Graph Neural Network model for predicting spatial counterfactuals, such as localised perturbations.

    Parameters
    ----------
    input_dim : int
        The dimensionality of the input gene expression features.
    output_dim : int
        The dimensionality of the output features after processing through graph convolution and linear layers.
    n_neighbors : int
        The number of neighbors to use in constructing the k-nearest neighbor graph.
    seed : int, optional
        Random seed for reproducibility, default is 0.

    Attributes
    ----------
    conv1 : GCNConv
        Graph convolutional layer for gene-to-gene (G2G) interactions.
    lin : torch.nn.Linear
        Linear layer for intracellular gene regulation.
    n_neighbors : int
        Number of spatial neighbors used for constructing the graph.
    sphex : torch.nn.Parameter or None
        Spherical gene expression matrix, set via `set_sphex`.
    gex : torch.nn.Parameter or None
        Gene expression matrix, calculated from the spherical expression matrix.
    output_dim : int
        Output dimensionality of the model.
    
    Methods
    -------
    set_g2g(g2g)
        Sets the gene-to-gene (G2G) interaction matrix artificially.
    set_g2g_intra(g2g_intra)
        Sets the intracellular gene regulation matrix artificially.
    set_sphex(sphex)
        Sets the spherical gene expression matrix artificially.
    forward(edge_index, batch)
        Forward pass of the model, calculating messages from gene-to-gene interactions, 
        intracellular interactions, and the log partition function (log(Z_mft)).
    log_Z_mft(edge_index, batch)
        Computes the Mean Field Theory (MFT) approximation to the partition function for the current gene expressions.
    z_interaction(num_spots, g)
        Calculates the interaction term for the partition function while avoiding numerical instability.
    calc_gex(sphex)
        Converts the spherical gene expression matrix into a regular gene expression matrix.
    calc_sphex(gex)
        Converts the regular gene expression matrix into a spherical gene expression matrix.
    get_pos(n_x, n_y)
        Generates a 2D hexagonal grid of positions for spatial modeling.
    normalize_g2g(g2g)
        Symmetrizes and normalizes the gene-to-gene interaction matrix.
    
    Examples
    --------
    >>> model = simcomen(input_dim=1000, output_dim=100, n_neighbors=6, seed=42)
    >>> edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    >>> batch = torch.tensor([0, 1], dtype=torch.long)
    >>> model.set_sphex(torch.randn(100, 1000))
    >>> msg, msg_intra, log_z_mft = model(edge_index, batch)
    >>> print(log_z_mft)
    """
    def __init__(self, input_dim, output_dim, n_neighbors, seed=0):
        """
        Initializes the `simcomen` model with a graph convolution layer and a linear layer 
        for gene-to-gene interactions and intracellular regulation, respectively.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input features.
        output_dim : int
            Dimensionality of the output features.
        n_neighbors : int
            Number of neighbors to use for constructing the spatial graph.
        seed : int, optional
            Random seed for reproducibility (default is 0).
        """
        super(simcomen, self).__init__()
        # define the seed
        torch.manual_seed(seed)
        # set up the graph convolution
        self.conv1 = GCNConv(input_dim, output_dim, add_self_loops=False)
        # set up the linear layer for intracellular gene regulation
        self.lin = torch.nn.Linear(input_dim, output_dim)
        # define the neighbors
        self.n_neighbors = n_neighbors
        # define a tracking variable for the gene expression x matrix
        self.sphex = None
        self.gex = None
        self.output_dim = output_dim
        self.edge_index = None

    # define a function to artificially set the g2g matrix
    def set_g2g(self, g2g):
        """
        Artificially sets the gene-to-gene (G2G) interaction matrix.

        Parameters
        ----------
        g2g : torch.Tensor
            A tensor representing gene-to-gene interactions to be used in the graph convolution.
        """
        # set the weight as the input
        self.conv1.lin.weight = torch.nn.Parameter(g2g, requires_grad=False)
        # and then set the bias as all zeros
        self.conv1.bias = torch.nn.Parameter(torch.from_numpy(np.zeros(self.output_dim).astype('float32')), requires_grad=False)

    # define a function to artificially set the g2g matrix
    def set_g2g_intra(self, g2g_intra):
        """
        Artificially sets the intracellular regulation matrix.

        Parameters
        ----------
        g2g_intra : torch.Tensor
            A tensor representing intracellular gene regulation interactions.
        """
        # set the weight as the input
        self.lin.weight = torch.nn.Parameter(g2g_intra, requires_grad=False)
        # and then set the bias as all zeros
        self.lin.bias = torch.nn.Parameter(torch.from_numpy(np.zeros(len(g2g_intra)).astype('float32')), requires_grad=False)

    # define a function to artificially set the sphex matrix
    def set_sphex(self, sphex):
        """
        Sets the spherical gene expression matrix for the forward pass.

        Parameters
        ----------
        sphex : torch.Tensor
            A tensor representing the spherical expression matrix.
        """
        self.sphex = torch.nn.Parameter(sphex, requires_grad=True)
        
    # define the forward pass
    def forward(self, edge_index, batch):
        """
        Forward pass of the model, calculates the messages between nodes using gene-to-gene interactions 
        and intracellular gene regulation. Also calculates the log partition function using Mean Field Theory.

        Parameters
        ----------
        edge_index : torch.Tensor
            Tensor representing the graph edges (connections between nodes/cells).
        batch : torch.Tensor
            Tensor representing the batch of data.

        Returns
        -------
        msg : torch.Tensor
            Message passed between nodes based on gene-to-gene interactions.
        msg_intra : torch.Tensor
            Message passed within nodes based on intracellular gene regulation.
        log_z_mft : torch.Tensor
            Mean Field Theory approximation of the log partition function.
        """
        # compute the gex
        self.gex = self.calc_gex(self.sphex)
        #print( f"self.gex device is {self.gex.device}")
        #print( f"edge_index device is {edge_index.device}")
        # compute the message
        msg = self.conv1(self.gex, edge_index)
        # compute intracellular message
        msg_intra = self.lin(self.gex)
        # compute the log z mft
        log_z_mft = self.log_Z_mft(edge_index, batch)
        return msg, msg_intra, log_z_mft

    # define approximation function
    def log_Z_mft(self, edge_index, batch):
        """
        Computes the Mean Field Theory (MFT) approximation of the partition function. 
        This function assumes that gene expression values are close to their mean across the spatial slide.

        Parameters
        ----------
        edge_index : torch.Tensor
            Tensor representing the graph edges (connections between nodes/cells).
        batch : torch.Tensor
            Tensor representing the batch of data.

        Returns
        -------
        log_z_mft : torch.Tensor
            Mean Field Theory approximation of the log partition function.
        """
        # retrieve number of spots
        num_spots = self.gex.shape[0]
        # calculate mean gene expression        
        mean_genes = torch.mean(self.gex, axis=0).reshape(-1,1)  # the mean should be per connected graph
        # calculate the norm of the sum of mean genes
        g = torch.norm(torch.mm( self.n_neighbors*self.conv1.lin.weight + 2*self.lin.weight, mean_genes)) 
        # calculate the contribution for mean values        
        z_mean = - num_spots  * torch.mm(torch.mm(torch.t(mean_genes), self.lin.weight + 0.5 * self.n_neighbors * self.conv1.lin.weight),  mean_genes)
        # calculate the contribution gene interactions
        z_interaction = self.z_interaction(num_spots=num_spots, g=g)
        # add the two contributions        
        log_z_mft = z_mean + z_interaction
        return log_z_mft

    def z_interaction(self, num_spots, g):
        """
        Calculates the interaction term for the partition function approximation, avoiding exploding exponentials.

        Parameters
        ----------
        num_spots : int
            Number of spots (cells) in the dataset.
        g : torch.Tensor
            Norm of the sum of mean gene expressions weighted by gene-to-gene interactions.

        Returns
        -------
        z_interaction : torch.Tensor
            Approximated interaction term for the partition function.
        """
        if g>20:
            z_interaction = num_spots * ( g - torch.log( g) )
        else:
            z_interaction = num_spots * torch.log((torch.exp( g) - torch.exp(- g))/( g))
        return z_interaction

    # define a function to derive the gex from the sphex
    def calc_gex(self, sphex):
        """
        Converts the spherical expression matrix into a regular gene expression matrix.

        Parameters
        ----------
        sphex : torch.Tensor
            The spherical gene expression matrix.

        Returns
        -------
        gex : torch.Tensor
            The converted regular gene expression matrix.
        """
        # setup the gex
        n_genes = sphex.shape[1]+1
        #gex = torch.from_numpy(np.zeros((sphex.shape[0], n_genes)).astype('float32'), device=next(self.parameters()).device)
        gex = torch.zeros((sphex.shape[0], n_genes), dtype=torch.float32, device=next(self.parameters()).device)
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
        Generates a 2D hexagonal grid of positions for spatial modelling.

        Parameters
        ----------
        n_x : int
            Number of positions along the x-axis.
        n_y : int
            Number of positions along the y-axis.

        Returns
        -------
        pos : numpy.ndarray
            Array of 2D positions for the grid.
        """
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
        Symmetrizes and normalizes the gene-to-gene interaction matrix.

        Parameters
        ----------
        g2g : numpy.ndarray
            The gene-to-gene interaction matrix.

        Returns
        -------
        g2g : numpy.ndarray
            The normalized and symmetrized gene-to-gene interaction matrix.
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
    
    # define a function to derive the gex from the sphex
    def calc_sphex(self, gex):
        """
        Converts the regular gene expression matrix into a spherical gene expression matrix.

        Parameters
        ----------
        gex : torch.Tensor
            The regular gene expression matrix.

        Returns
        -------
        sphex : torch.Tensor
            The converted spherical gene expression matrix.
        """
        # setup the gex
        n_sgenes = gex.shape[1]-1
        #sphex = torch.from_numpy(np.zeros((gex.shape[0], n_sgenes)).astype('float32'), device=next(self.parameters()).device)
        sphex = torch.zeros((gex.shape[0], n_sgenes), dtype=torch.float32, device=next(self.parameters()).device)
        # compute the gex
        for idx in range(n_sgenes):
            sphex[:,idx] = gex[:,idx]
            for idx_ in range(idx):
                sphex[:,idx] /= torch.sin(sphex[:,idx_])
            sphex[:,idx] = torch.arccos(sphex[:,idx])
        return sphex
    
    # -------------------------
    # Propagation-based generation (no training)
    # -------------------------
    def set_edge_index(self, edge_index: torch.Tensor):
        """
        Set graph connectivity to be used for propagation.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge list in COO format with shape (2, E).
        """
        self.edge_index = edge_index

    def apply_network_propagation(
        self,
        initial_expression: torch.Tensor,
        perturbation_mask_expression: Optional[torch.Tensor] = None,
        perturbation_values_expression: Optional[torch.Tensor] = None,
        scale_factor_gex: float = 0.5,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Apply network propagation over the cell graph using the model's learned weights.

        Parameters
        ----------
        initial_expression : torch.Tensor
            Initial gene expression matrix of shape (num_cells, num_genes).
        perturbation_mask_expression : Optional[torch.Tensor], optional
            Boolean mask of shape (num_cells, num_genes) indicating perturbed entries, by default None.
        perturbation_values_expression : Optional[torch.Tensor], optional
            Values for the perturbations, same shape as the mask/value tensor, by default None.
        scale_factor_gex : float, optional
            Scaling factor applied to propagated messages, by default 0.5.
        normalize : bool, optional
            If True, L2-normalize expression per cell after propagation, by default True.

        Returns
        -------
        torch.Tensor
            Updated (and optionally normalized) expression matrix of shape (num_cells, num_genes).
        """
        if self.edge_index is None:
            raise ValueError("edge_index must be set via set_edge_index before propagation")

        num_cells, num_genes = initial_expression.shape

        # Ensure dimensions are consistent with square weights (common in this codebase)
        conv_weight = self.conv1.lin.weight  # shape: (out_dim, in_dim)
        lin_weight = self.lin.weight         # shape: (out_dim, in_dim)
        if not (conv_weight.shape[0] == conv_weight.shape[1] == num_genes and
                lin_weight.shape[0] == lin_weight.shape[1] == num_genes):
            raise ValueError(
                "Propagation expects square weight matrices matching the number of genes. "
                f"Got conv {tuple(conv_weight.shape)}, lin {tuple(lin_weight.shape)}, genes {num_genes}."
            )

        device = initial_expression.device
        current_expression = initial_expression.clone()

        # Build dense adjacency from edge_index
        adjacency_matrix = torch.zeros((num_cells, num_cells), dtype=initial_expression.dtype, device=device)
        sources = self.edge_index[0]
        targets = self.edge_index[1]
        adjacency_matrix[sources, targets] = 1.0
        adjacency_matrix.fill_diagonal_(0)

        # Initialize delta with explicit perturbations, if any
        delta_expression = torch.zeros_like(initial_expression)
        if perturbation_mask_expression is not None and perturbation_values_expression is not None:
            delta_expression[perturbation_mask_expression] = perturbation_values_expression[perturbation_mask_expression]

        # Single-step propagation using explicit perturbations
        effective_delta = delta_expression

        # Intercellular message: neighbors then apply intercellular G2G weights
        intercellular_gene_gene = adjacency_matrix.mm(effective_delta).mm(conv_weight.t())
        # Intracellular message: within-cell linear regulation
        intracellular_gene_gene = effective_delta.mm(lin_weight.t())

        # Update expression
        current_expression = current_expression + (intercellular_gene_gene + intracellular_gene_gene) * scale_factor_gex

        if normalize:
            # L2-normalize per cell, avoid division by zero
            norm = torch.linalg.norm(current_expression, dim=1, keepdim=True)
            norm = torch.clamp(norm, min=1e-8)
            current_expression = current_expression / norm

        return current_expression

    def forward_with_propagation(
        self,
        edge_index: torch.Tensor,
        gene_expression: torch.Tensor,
        perturbation_mask_expression: Optional[torch.Tensor] = None,
        perturbation_values_expression: Optional[torch.Tensor] = None,
        scale_factor_gex: float = 0.5,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass that performs network propagation (no training) using the model's weights.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge list in COO format with shape (2, E).
        gene_expression : torch.Tensor
            Initial gene expression matrix of shape (num_cells, num_genes).
        perturbation_mask_expression : Optional[torch.Tensor], optional
            Boolean mask for perturbations, by default None.
        perturbation_values_expression : Optional[torch.Tensor], optional
            Values for perturbations, by default None.
        scale_factor_gex : float, optional
            Scaling factor applied to propagated messages, by default 0.5.
        normalize : bool, optional
            If True, L2-normalize expression per cell after propagation, by default True.

        Returns
        -------
        torch.Tensor
            Propagated (and optionally normalized) expression matrix.
        """
        self.set_edge_index(edge_index)
        updated_expression = self.apply_network_propagation(
            initial_expression=gene_expression,
            perturbation_mask_expression=perturbation_mask_expression,
            perturbation_values_expression=perturbation_values_expression,
            scale_factor_gex=scale_factor_gex,
            normalize=normalize,
        )
        return updated_expression


# # Assume `sim` is a trained `simcomen` or has weights set via set_g2g/set_g2g_intra
# # edge_index: (2, E) tensor
# # gene_expression: (num_cells, num_genes) tensor, normalized if desired

# # Optional perturbation
# perturb_mask = torch.zeros_like(gene_expression, dtype=torch.bool)
# perturb_vals = torch.zeros_like(gene_expression)
# # e.g., knock down gene j in cell i:
# # perturb_mask[i, j] = True
# # perturb_vals[i, j] = -gene_expression[i, j]

# updated_gex = sim.forward_with_propagation(
#     edge_index=edge_index,
#     gene_expression=gene_expression,
#     perturbation_mask_expression=perturb_mask,
#     perturbation_values_expression=perturb_vals,
#     scale_factor_gex=0.5,
#     num_steps=1,
#     normalize=True,
# )