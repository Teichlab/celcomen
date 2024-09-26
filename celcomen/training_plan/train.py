from tqdm import tqdm
import numpy as np
import torch
from ..utils.helpers import normalize_g2g, calc_sphex, calc_gex

def train(epochs, learning_rate, model, loader, zmft_scalar=1e-1, seed=1, device="cpu", verbose=False):
    
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
            model.conv1.lin.weight = torch.nn.Parameter(normalize_g2g(model.conv1.lin.weight), requires_grad=True)
            model.lin.weight = torch.nn.Parameter(normalize_g2g(model.lin.weight), requires_grad=True)
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
            
        if verbose: print(f"Epoch={epoch}   |   Loss={np.mean(losses_)}")
        losses.append(np.mean(losses_))

    return losses


def train_simcomen(epochs, learning_rate, model, edge_index, zmft_scalar=1e-1, seed=1, device="cpu", verbose=False):
    
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










