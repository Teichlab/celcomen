from torch_geometric.nn import GCNConv
from sklearn.neighbors import kneighbors_graph
import torch
import numpy as np
#from ..utils.helpers import calc_gex

# define the number of neighbors (six for visium)
n_neighbors = 6
# define the simcomen class
class simcomen(torch.nn.Module):
    # define initialization function
    def __init__(self, input_dim, output_dim, n_neighbors, seed=0):
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

    # define a function to artificially set the g2g matrix
    def set_g2g(self, g2g):
        """
        Artifically sets the core g2g matrix to be a specified interaction matrix
        """
        # set the weight as the input
        self.conv1.lin.weight = torch.nn.Parameter(g2g, requires_grad=False)
        # and then set the bias as all zeros
        self.conv1.bias = torch.nn.Parameter(torch.from_numpy(np.zeros(self.output_dim).astype('float32')), requires_grad=False)

    # define a function to artificially set the g2g matrix
    def set_g2g_intra(self, g2g_intra):
        """
        Artifically sets the core g2g intracellular matrix to be a specified matrix
        """
        # set the weight as the input
        self.lin.weight = torch.nn.Parameter(g2g_intra, requires_grad=False)
        # and then set the bias as all zeros
        self.lin.bias = torch.nn.Parameter(torch.from_numpy(np.zeros(len(g2g_intra)).astype('float32')), requires_grad=False)

    # define a function to artificially set the sphex matrix
    def set_sphex(self, sphex):
        """
        Artifically sets the current sphex matrix
        """
        self.sphex = torch.nn.Parameter(sphex, requires_grad=True)
        
    # define the forward pass
    def forward(self, edge_index, batch):
        """
        Forward pass for prediction or training,
        convolutes the input by the expected interactions and returns log(Z_mft)
        """
        # compute the gex
        self.gex = calc_gex(self.sphex)
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
        Mean Field Theory approximation to the partition function. Assumptions used are:
        - expression of values of genes are close to their mean values over the visium slide
        - \sum_b g_{a,b} m^b >0 \forall a, where m is the mean gene expression and g is the gene-gene
          interaction matrix.
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
        Avoid exploding exponentials by returning an approximate interaction term for the partition function.
        """
        if g>20:
            z_interaction = num_spots * ( g - torch.log( g) )
        else:
            z_interaction = num_spots * torch.log((torch.exp( g) - torch.exp(- g))/( g))
        return z_interaction


    # define a function to derive the gex from the sphex
    def calc_gex(sphex):
        """
        Calculates the gene expression matrix from the spherical
        """
        # setup the gex
        n_genes = sphex.shape[1]+1
        gex = torch.from_numpy(np.zeros((sphex.shape[0], n_genes)).astype('float32'), device=next(model.parameters()).device)
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
        Addresses any small fluctuations in symmetrical weights
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
    def calc_sphex(gex):
        """
        Calculates the spherical expression matrix from the normal
        """
        # setup the gex
        n_sgenes = gex.shape[1]-1
        sphex = torch.from_numpy(np.zeros((gex.shape[0], n_sgenes)).astype('float32'), device=next(model.parameters()).device)
        # compute the gex
        for idx in range(n_sgenes):
            sphex[:,idx] = gex[:,idx]
            for idx_ in range(idx):
                sphex[:,idx] /= torch.sin(sphex[:,idx_])
            sphex[:,idx] = torch.arccos(sphex[:,idx])
        return sphex
    
