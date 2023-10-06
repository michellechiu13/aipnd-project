import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type = str, default='.')
    parser.add_argument('--arch', type = str, default="vgg16")
    parser.add_argument('--learning_rate', type = float, default=0.001)
    parser.add_argument('--hidden_units', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--gpu', action='store_true')
    return parser.parse_args()

def load_data():
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    val_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_data = datasets.ImageFolder(valid_dir, transform=val_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    return train_data, val_data, test_data, trainloader, valloader, testloader

def build_model(arch, hidden_unit):
    if arch == "vgg11":
        model = models.vgg11(weights=models.VGG11_Weights.DEFAULT)
        in_features = model.classifier[0].in_features
    if arch == "vgg13":
        model = models.vgg13(weights=models.VGG13_Weights.DEFAULT)
        in_features = model.classifier[0].in_features
    elif arch == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        in_features = model.classifier[0].in_features

    for param in model.parameters():
        param.requires_grad = False

    
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(in_features, hidden_unit)),
                            ('relu', nn.ReLU()),
                            ('fc2', nn.Linear(hidden_unit, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    if arch == "vgg11" or arch == "vgg16":
        model.classifier = classifier
    else:
        model.fc = classifier
    return model

def train_model(epochs, trainloader,  valloader, device, model, optimizer, criterion):
    steps = 0
    running_loss = 0
    running_accuracy = 0
    print_every = 5

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            model.train()
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        test_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Val loss: {test_loss/len(valloader):.3f}.. "
                    f"Val accuracy: {accuracy/len(valloader):.3f}.."
                    f"Train accuracy: {running_accuracy/print_every:.3f}")
                running_loss = 0
                running_accuracy = 0

def save_checkpoint(model):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
                  'classifier' : model.classifier,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                 }

    torch.save(checkpoint, 'checkpoint.pth')

if __name__ == "__main__":

    args = get_input_args()

    learning_rate = args.learning_rate
    hidden_unit = args.hidden_units
    arch = args.arch
    epochs = args.epochs

    train_data, _, _,  trainloader, valloader, testloader = load_data()


    model = build_model(arch, hidden_unit)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    if arch == "vgg11" or arch == "vgg16":
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    train_model(epochs, trainloader, valloader, device, model, optimizer, criterion)

    model.class_to_idx = train_data.class_to_idx
    save_checkpoint(model)