import torch
from torch_geometric.nn import GCNConv
from sklearn.neighbors import kneighbors_graph
import numpy as np

# define the celcomen class
class celcomen(torch.nn.Module):
    """
    A neural network model for gene-to-gene (G2G) and intracellular regulation, 
    built on graph convolutions. The model also supports a mean-field theory (MFT) approximation 
    to estimate the partition function, integrating spatial and gene expression data.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features (gene expression data).
    output_dim : int
        Dimensionality of the output features.
    n_neighbors : int
        The number of neighbours used in the spatial graph to model cell-cell interactions.
    seed : int, optional
        Seed for random number generation to ensure reproducibility. Default is 0.

    Attributes
    ----------
    conv1 : GCNConv
        A graph convolutional layer that models gene-to-gene interactions (G2G).
    lin : torch.nn.Linear
        A linear layer that models intracellular gene regulation.
    n_neighbors : int
        The number of neighbours for spatial graph construction.
    gex : torch.nn.Parameter or None
        Stores the gene expression matrix used for the forward pass. Set to None initially.
    
    Methods
    -------
    set_g2g(g2g)
        Sets the gene-to-gene (G2G) interaction matrix artificially.
    set_g2g_intra(g2g_intra)
        Sets the intracellular regulation matrix artificially.
    set_gex(gex)
        Sets the gene expression matrix artificially.
    forward(edge_index, batch)
        Forward pass to compute the gene-to-gene and intracellular messages, 
        and the log partition function estimate.
    log_Z_mft(edge_index, batch)
        Computes the Mean Field Theory (MFT) approximation to the partition function.
    z_interaction(num_spots, g)
        Provides an approximation for the interaction term in the partition function 
        to prevent numerical instability due to exploding exponentials.

    Examples
    --------
    >>> model = celcomen(input_dim=1000, output_dim=100, n_neighbors=6, seed=42)
    >>> edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    >>> batch = torch.tensor([0, 1], dtype=torch.long)
    >>> model.set_gex(torch.randn(100, 1000))
    >>> msg, msg_intra, log_z_mft = model(edge_index, batch)
    >>> print(log_z_mft)
    """
    
    def __init__(self, input_dim, output_dim, n_neighbors, seed=0):
        """
        Initializes the celcomen model with a graph convolution layer and a linear 
        layer for gene-to-gene and intracellular regulation, respectively.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input features.
        output_dim : int
            Dimensionality of the output features.
        n_neighbors : int
            Number of neighbours for constructing the spatial graph.
        seed : int, optional
            Random seed for reproducibility (default is 0).
        """
        super(celcomen, self).__init__()
        # define the seed
        torch.manual_seed(seed)
        # set up the graph convolution
        self.conv1 = GCNConv(input_dim, output_dim, add_self_loops=False)
        # set up the linear layer for intracellular gene regulation
        self.lin = torch.nn.Linear(input_dim, output_dim)
        # define the neighbors
        self.n_neighbors = n_neighbors
        # define a tracking variable for the gene expression x matrix
        self.gex = None

    # define a function to artificially set the g2g matrix
    def set_g2g(self, g2g):
        """
        Artificially sets the gene-to-gene (G2G) interaction matrix.

        Parameters
        ----------
        g2g : torch.Tensor
            A matrix representing gene-to-gene interactions to be used for graph convolution.
        """
        # set the weight as the input
        self.conv1.lin.weight = torch.nn.Parameter(g2g, requires_grad=True)
        # and then set the bias as all zeros
        self.conv1.bias = torch.nn.Parameter(torch.from_numpy(np.zeros(len(g2g)).astype('float32')), requires_grad=False)

    # define a function to artificially set the g2g matrix
    def set_g2g_intra(self, g2g_intra):
        """
        Artificially sets the intracellular gene regulation matrix.

        Parameters
        ----------
        g2g_intra : torch.Tensor
            A matrix representing intracellular gene regulation interactions.
        """
        # set the weight as the input
        self.lin.weight = torch.nn.Parameter(g2g_intra, requires_grad=True)
        # and then set the bias as all zeros
        self.lin.bias = torch.nn.Parameter(torch.from_numpy(np.zeros(len(g2g_intra)).astype('float32')), requires_grad=False)

    # define a function to artificially set the sphex matrix
    def set_gex(self, gex):
        """
        Sets the gene expression matrix to be used during the forward pass.

        Parameters
        ----------
        gex : torch.Tensor
            A matrix representing the gene expression of the cells.
        """
        self.gex = torch.nn.Parameter(gex, requires_grad=False)
        
    # define the forward pass
    def forward(self, edge_index, batch):
        """
        Forward pass for the model, computing gene-to-gene and intracellular messages, 
        and estimating the log partition function using Mean Field Theory (MFT).

        Parameters
        ----------
        edge_index : torch.Tensor
            Tensor representing the graph edges (connectivity between nodes/cells).
        batch : torch.Tensor
            Tensor representing the batch of data.

        Returns
        -------
        msg : torch.Tensor
            The message propagated between cells based on gene-to-gene interactions.
        msg_intra : torch.Tensor
            The message based on intracellular gene regulation.
        log_z_mft : torch.Tensor
            The Mean Field Theory approximation to the log partition function.
        """
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
        Computes the Mean Field Theory (MFT) approximation to the partition function, 
        which estimates the likelihood of gene expression states in the dataset.

        Parameters
        ----------
        edge_index : torch.Tensor
            Tensor representing the graph edges (connectivity between nodes/cells).
        batch : torch.Tensor
            Tensor representing the batch of data.

        Returns
        -------
        log_z_mft : torch.Tensor
            The log partition function estimated using Mean Field Theory (MFT).
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
        Avoids exploding exponentials in the partition function approximation by 
        returning an approximate interaction term.

        Parameters
        ----------
        num_spots : int
            Number of spots (cells) in the dataset.
        g : torch.Tensor
            Norm of the sum of mean gene expressions weighted by gene-to-gene interactions.

        Returns
        -------
        z_interaction : torch.Tensor
            The approximated interaction term for the partition function.
        """
        if g>20:
            z_interaction = num_spots * ( g - torch.log( g) )
        else:
            z_interaction = num_spots * torch.log((torch.exp( g) - torch.exp(- g))/( g))
        return z_interaction