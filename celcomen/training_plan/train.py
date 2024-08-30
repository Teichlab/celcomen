from tqdm import tqdm
import numpy as np
import torch

# def train(model, optimizer, train_loader, num_epochs, seed=1, device="cpu"):
    
#     if device is None:
#         device = torch.device(device)
    
#     torch.manual_seed(seed)
    
#     for epoch in range(num_epochs):
        
#         model = model.train()
#         for batch_idx, (features, labels) in enumerate(train_loader):
            
#             features, labels = features.to(device),  labels.to(device)
            
#             logits = model(features)
            
#             loss = F.cross_entropy(logits, labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
            
def train(num_epochs, learning_rate, model, loader, seed=1, device="cpu"):
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
    losses = []
    model.train()
    torch.manual_seed(seed)
    
    for epoch in range(num_epochs):
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
            losses_.append(loss.detach().numpy()[0][0])
            # derive the gradients, update, and clear
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # repeatedly force a normalization
            model.conv1.lin.weight = torch.nn.Parameter(normalize_g2g(model.conv1.lin.weight), requires_grad=True)
            model.lin.weight = torch.nn.Parameter(normalize_g2g(model.lin.weight), requires_grad=True)
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
            
        print(f"Loss={np.mean(losses_)}")
        losses.append(np.mean(losses_))

    return losses


