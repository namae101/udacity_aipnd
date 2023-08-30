import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def get_input_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str, default='flowers', help='path to the folder of flower images')
    parser.add_argument('--save_dir', type=str, default='.', help='path to save the checkpoint')
    parser.add_argument('--arch', type=str, default='vgg11', help='architecture of the model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=256, help='number of hidden units')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--gpu', action='store_true', help='use GPU for training')

    return parser.parse_args()


def load_and_freezed_pretrained_model(arch):
    if arch == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    else:
        print('Please choose either vgg11 or resnet18')
        return None

    for param in model.parameters():
        param.requires_grad = False

    return model

def build_classifier(arch,model, input_size, hidden_units):
    classifier = nn.Sequential(nn.Linear(input_size, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(hidden_units, 102),
                               nn.LogSoftmax(dim=1))
    if arch == 'vgg11':
        model.classifier = classifier
    elif arch == 'resnet18':
        model.fc = classifier
    return model

#Return Data Loader
def data_loader(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(256),
                                       transforms.RandomHorizontalFlip(),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    test_validation_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_validation_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    return train_loader, valid_loader, test_loader, train_data.class_to_idx


def training(model, train_loader, valid_loader, criterion, optimizer, epochs, device):
    model.to(device)
    steps = 0
    running_loss = 0
    step_every_validation = 5
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % step_every_validation == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
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
                      f"Step {steps+1}.. "
                      f"Average Train loss Each Step: {running_loss/step_every_validation:.3f}.. "
                      f"Validation loss: {test_loss/len(valid_loader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(valid_loader)*100:.3f}%")
                running_loss = 0
                model.train()


def save_checkpoint(model, class_to_idx, optimizer, in_arg):
    checkpoint = {'model': model,
                  'class_to_idx': class_to_idx,}
    torch.save(checkpoint, in_arg.save_dir + '/checkpoint.pth')

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    print('Test loss: {:.3f}.. '.format(test_loss/len(test_loader)),
          'Test accuracy: {:.3f}%'.format(accuracy/len(test_loader)*100))

def main():
    in_arg = get_input_args()
    
    device = torch.device('cpu')
    if in_arg.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == 'cpu':
            print('GPU is not available, use CPU instead for training')
        else:
            print('GPU is available, Using GPU for training')
    else:
        print('Using CPU for training')


    valid_arch = ['vgg11', 'resnet18']
    if in_arg.arch not in valid_arch:
        print('Please choose either vgg11 or resnet18')
        return None
    try:
        train_loader, valid_loader, test_loader, class_to_idx = data_loader(in_arg.data_dir)
    except:
        print('Please check the data directory')
        return None

    # Load pretrained model, freezed the parameters and build classifier
    if in_arg.arch == 'vgg11':
        model = models.vgg11(pretrained=True)
        input_size = 25088
        model = build_classifier(in_arg.arch, model, input_size, in_arg.hidden_units)
    elif in_arg.arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_size = 512
        model = build_classifier(in_arg.arch, model, input_size, in_arg.hidden_units)
    

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    if in_arg.arch == 'vgg11':
        optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)
    elif in_arg.arch == 'resnet18':
        optimizer = optim.Adam(model.fc.parameters(), lr=in_arg.learning_rate)


    print('Start training for {} epochs with learning rate {}'.format(in_arg.epochs, in_arg.learning_rate))
    # Training
    training(model, train_loader, valid_loader, criterion, optimizer, in_arg.epochs, device)

    print('Training completed')

    # Testing Modeling
    print('Start testing')
    test(model, test_loader, criterion, device)
    print('Testing completed')


    # Save the checkpoint
    save_checkpoint(model,class_to_idx, optimizer, in_arg)
    print('Checkpoint saved to {}'.format(in_arg.save_dir + '/checkpoint.pth'))

    


# Call to main function to run the program
if __name__ == "__main__":
    main()