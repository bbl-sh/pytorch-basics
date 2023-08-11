import time
import torch
import torch.nn.functional as F

def train_model(num_epochs, model, train_loader, test_loader,optimizer):
    for epoch in range(num_epochs):
      model.train()
      for batch_idx, (feature, target) in enumerate(train_loader):
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        if not batch_idx % 5000:
            print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                    f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                    f'| Loss: {loss:.4f}')
