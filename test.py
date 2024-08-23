import torch


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
    test_loss /= len(test_loader.dataset)

    print(f"Test Loss: {test_loss:.6f}")


if __name__ == "__main__":
    test(model, device, test_loader, criterion)
