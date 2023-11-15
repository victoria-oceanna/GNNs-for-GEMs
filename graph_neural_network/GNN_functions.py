import torch

def train(model, train_loader, test_loader, device, criterion, optimizer):
    model.train()

    for data in train_loader: 
        data.to(device)
        features = data.x.to(torch.float32)
        out = model(features, data.edge_index, data.batch)
        loss = criterion(out, data.y)  
        loss.backward()
        optimizer.step()  
        optimizer.zero_grad() 

def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data.to(device)
        features = data.x.to(torch.float32)
        out = model(features, data.edge_index, data.batch)  
        pred = out.argmax(dim=1) 
        correct += int((pred == data.y).sum())  
    return correct / len(loader.dataset) 