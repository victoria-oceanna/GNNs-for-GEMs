import torch

def train(model, train_loader, device, criterion, optimizer):
     """Trains GNN model

    :param model: GNN model to train
    :param train_loader: Dataloader for training portion of dataset
    :param device: CUDA device for GPU computing
    :param criterion: Training criterion
    :param optimizer: Optimization algorithm
    :return: None
    """
    model.train()

    for data in train_loader: 
        data.to(device)
        features = data.x.to(torch.float32)
        out = model(features, data.edge_index, data.batch)
        loss = criterion(out, data.y)  
        loss.backward()
        optimizer.step()  
        optimizer.zero_grad() 

def test(model, test_loader, device):
    """Applies GNN model to test dataset and evaluates performance

    :param model: Trained GNN model to test
    :param train_loader: Dataloader for test portion of dataset
    :param device: CUDA device for GPU computing
    :return: Model accuracy (0-1)
    """
    model.eval()
    correct = 0
    for data in test_loader:
        data.to(device)
        features = data.x.to(torch.float32)
        out = model(features, data.edge_index, data.batch)  
        pred = out.argmax(dim=1) 
        correct += int((pred == data.y).sum())  
    return correct / len(test_loader.dataset) 
