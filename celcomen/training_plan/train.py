from tqdm import tqdm
import numpy as np
import torch
from ..utils.helpers import symmetrize_g2g, calc_sphex, calc_gex

def train(epochs, learning_rate, model, loader, zmft_scalar=1e-1, seed=1, device="cpu", verbose=False):
    """
    Trains the model using a stochastic gradient descent (SGD) optimizer over the specified number of epochs.

    During training, the model calculates messages between genes based on gene-to-gene (G2G) interactions 
    and applies a Mean Field Theory (MFT) approximation for gene expression interactions. 
    The G2G and intracellular regulation matrices are normalized after each step.

    Parameters
    ----------
    epochs : int
        Number of training epochs to run.
    learning_rate : float
        Learning rate for the SGD optimizer.
    model : torch.nn.Module
        The model to be trained, which includes a graph convolutional layer and a linear layer.
    loader : torch_geometric.loader.DataLoader
        DataLoader that provides the data for each batch during training.
    zmft_scalar : float, optional
        Scalar to weight the Mean Field Theory term in the loss function. Default is 1e-1.
    seed : int, optional
        Seed for random number generation to ensure reproducibility. Default is 1.
    device : str, optional
        Device to use for training, e.g., "cpu" or "cuda". Default is "cpu".
    verbose : bool, optional
        If True, prints the loss at each epoch. Default is False.

    Returns
    -------
    losses : list of float
        List of losses recorded at each epoch.

    Examples
    --------
    >>> losses = train(epochs=100, learning_rate=0.01, model=my_model, loader=my_loader, zmft_scalar=1e-1, seed=42, device="cuda")
    >>> print(losses[-1])  # Final loss
    """    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
    losses = []
    model.train()
    torch.manual_seed(seed)
    
    for epoch in tqdm(range(epochs), total=epochs):
        losses_= []

        for data in loader:
            # move data to device
            data = data.to(device)
            # train loader  # Iterate in batches over the training dataset.
            # set the appropriate gex
            model.set_gex(data.x)
            # derive the message as well as the mean field approximation
            msg, msg_intra, log_z_mft = model(data.edge_index, 1)
            # compute the loss and track it
            loss = -(-log_z_mft + zmft_scalar * torch.trace(torch.mm(msg, torch.t(model.gex))) + zmft_scalar * torch.trace(torch.mm(msg_intra, torch.t(model.gex))) )
            if device=="cpu":
                losses_.append(loss.detach().numpy()[0][0])
            else:
                losses_.append(loss.detach().cpu().numpy()[0][0])
            # derive the gradients, update, and clear
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # repeatedly force a normalization
            model.conv1.lin.weight = torch.nn.Parameter(symmetrize_g2g(model.conv1.lin.weight), requires_grad=True)
            model.lin.weight = torch.nn.Parameter(symmetrize_g2g(model.lin.weight), requires_grad=True)
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
            
        if verbose: print(f"Epoch={epoch}   |   Loss={np.mean(losses_)}")
        losses.append(np.mean(losses_))

    return losses


def train_simcomen(epochs, learning_rate, model, edge_index, zmft_scalar=1e-1, seed=1, device="cpu", verbose=False):
    """
    Trains the `simcomen` model using stochastic gradient descent (SGD) over the specified number of epochs.

    The training process computes the Mean Field Theory (MFT) approximation of the partition function
    and normalizes gene expression data using a spherical representation. The loss is tracked and updated 
    based on both gene-to-gene (G2G) and intracellular interactions.

    Parameters
    ----------
    epochs : int
        Number of training epochs to run.
    learning_rate : float
        Learning rate for the SGD optimizer.
    model : torch.nn.Module
        The `simcomen` model to be trained, which includes graph convolution and linear layers.
    edge_index : torch.Tensor
        Tensor representing the edges in the graph, i.e., the connections between nodes (cells).
    zmft_scalar : float, optional
        Scalar to weight the Mean Field Theory term in the loss function. Default is 1e-1.
    seed : int, optional
        Seed for random number generation to ensure reproducibility. Default is 1.
    device : str, optional
        Device to use for training, e.g., "cpu" or "cuda". Default is "cpu".
    verbose : bool, optional
        If True, prints the loss at each epoch. Default is False.

    Returns
    -------
    losses : list of float
        List of losses recorded at each epoch.

    Examples
    --------
    >>> losses = train_simcomen(epochs=100, learning_rate=0.01, model=my_model, edge_index=my_edge_index, zmft_scalar=1e-1, seed=42, device="cuda")
    >>> print(losses[-1])  # Final loss
    """    
    # set up the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
    # keep track of the losses per data object
    losses = []
    model.train()
    torch.manual_seed(seed)

    tmp_gexs = []
    # work through epochs
    for epoch in tqdm(range(epochs), total=epochs):
        # derive the message as well as the mean field approximation
        msg, msg_intra, log_z_mft = model(edge_index, 1)
        if (epoch % 5) == 0:
            if device=="cpu":
                tmp_gex = model.gex.clone().detach().numpy()
            else:
                tmp_gex = model.gex.clone().detach().cpu().numpy()
            tmp_gexs.append(tmp_gex)
        # compute the loss and track it
        loss = -(-log_z_mft + zmft_scalar * torch.trace(torch.mm(msg, torch.t(model.gex))) + zmft_scalar * torch.trace(torch.mm(msg_intra, torch.t(model.gex))) )
        if device=="cpu":
            losses.append(loss.detach().numpy()[0][0])
        else:
            losses.append(loss.detach().cpu().numpy()[0][0])
        # derive the gradients, update, and clear
        if verbose: print(f"Epoch={epoch}   |   Loss={np.mean(losses[-1])}")
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return losses










