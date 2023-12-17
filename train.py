import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import CNNStudent, CNNTeacher, Resnet18, Resnet34, Resnet50
import wandb
import argparse
from time import perf_counter

# function to extract cifar-10 datasets and apply transformations
def get_cifar10_datasets(img_size=32):
    # perform these data tx to augment data
    # normalization constants suggested by creators of cifar10
    train_transforms_cifar = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.1),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False)
    ])

    test_transforms_cifar = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load and download data
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms_cifar)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms_cifar)

    return train_dataset, test_dataset

# function to train model for 1 epoch
train_batch_counter = 0
def train(model, criterion, optimizer, train_loader, device, log_wandb):
    model.train()

    running_loss = 0.0
    for inputs, labels in train_loader:
        # inputs are a batch of images
        # labels are a batch of vectors representing the class of each image (one-hot encoded)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs)

        # outputs: Output of the network for the collection of images. A tensor of dimensionality batch_size x num_classes
        # labels: The actual labels of the images. Vector of dimensionality batch_size
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if log_wandb:
            global train_batch_counter
            wandb.log({"batch_train_loss": loss.item(), "batch_train": train_batch_counter})
            train_batch_counter += 1
        running_loss += loss.item()

    return running_loss / len(train_loader)

# function to evaluate model for 1 epoch
test_batch_counter = 0
def test(model, test_loader, device, log_wandb):
    model.eval()

    correct = 0
    total = 0

    # not training parameters
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            batch_total = labels.size(0)
            batch_correct = (predicted == labels).sum().item()
            if log_wandb:
                global test_batch_counter
                wandb.log({"batch_test_acc": 100 * batch_correct / batch_total, "batch_test": test_batch_counter})
                test_batch_counter += 1
            total += batch_total
            correct += batch_correct

    accuracy = 100 * correct / total
    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='Device to use for training.', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--train_batch_size', type=int, help='Batch size to use for training dataset.', default=128)
    parser.add_argument('--test_batch_size', type=int, help='Batch size to use for testing dataset.', default=128)
    parser.add_argument('--train_num_workers', type=int, help='Number of workers to use for training DataLoader.', default=2)
    parser.add_argument('--test_num_workers', type=int, help='Number of workers to use for testing DataLoader.', default=2)
    parser.add_argument('--epochs', type=int, help='Number of epochs to train for.', default=10)
    parser.add_argument('--lr', type=float, help='Learning rate to use for the optimizer.', default=0.001)
    parser.add_argument('--dropout', type=float, help='Dropout rate to use for the model.', default=0.1)
    parser.add_argument('--loss', help='Type of loss function to use during training.', default='ce', choices=['ce'])
    parser.add_argument('--opt', help='Type of optimizer to use during training.', default='adam', choices=['adam'])
    parser.add_argument('--seed', help='PyTorch seed to use for reproducability.', default=None, type=int)
    parser.add_argument('--mod', help='Name of model to train.', choices=['cnn_teacher', 'cnn_student', 'resnet18_student', 'resnet34_teacher', 'resnet50_teacher', 'resnet18_pt_student', 'resnet34_pt_teacher', 'resnet50_pt_teacher'])
    parser.add_argument('--verbose', help='Whether or not to print debugging outputs.', type=int, default=1)
    parser.add_argument('--log_wandb', help='Whether or not to log metrics to Wandb.', type=int, default=0)
    parser.add_argument('--mod_save_path', help='Path of where to save the trained model.', default=None)
    args = parser.parse_args()

    if args.seed:
        torch.manual_seed(args.seed)
    
    # load data
    print('Loading data...')
    if 'resnet' in args.mod:
        img_size = 224
    elif 'cnn' in args.mod:
        img_size = 32
    train_data, test_data = get_cifar10_datasets(img_size=img_size)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, num_workers=args.train_num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.test_num_workers)
    if args.verbose:
        print(f'Number of training examples: {len(train_loader)}')
        print(f'Number of testing examples: {len(test_loader)}')
    print('Done loading data!')

    # initialize model
    if args.verbose: print('Initializing model...')
    start = perf_counter()
    if args.mod == 'cnn_teacher':
        mod = CNNTeacher(num_classes=10, dropout=args.dropout)
    elif args.mod == 'cnn_student':
        mod = CNNStudent(num_classes=10, dropout=args.dropout)
    elif args.mod == 'resnet18_student':
        mod = Resnet18(pretrained=False)
    elif args.mod == 'resnet34_teacher':
        mod = Resnet34(pretrained=False)
    elif args.mod == 'resnet50_teacher':
        mod = Resnet50(pretrained=False)
    elif args.mod == 'resnet18_pt_student':
        mod = Resnet18(pretrained=True)
    elif args.mod == 'resnet34_pt_teacher':
        mod = Resnet34(pretrained=True)
    elif args.mod == 'resnet50_pt_teacher':
        mod = Resnet50(pretrained=True)
    
    mod = mod.to(args.device)
    end = perf_counter()
    mod_init_time = end - start
    if args.verbose: print('Done initializing model!')

    if args.verbose: print('Initializing model...')
    # initialize optimizer
    start = perf_counter()
    if args.opt == 'adam':
        optimizer = optim.Adam(mod.parameters(), lr=args.lr)
    end = perf_counter()
    opt_init_time = end - start

    # define loss
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    
    # init wandb to log training metrics
    if args.log_wandb:
        wandb.init(
                # set the wandb project where this run will be logged
                project="knowledge-distillation-experiments",
                
                # track hyperparameters and run metadata
                config={
                "exp_name": 'cifar10_model_training',
                "epochs": args.epochs,
                "train_data_size": len(train_loader.dataset),
                "test_data_size": len(test_loader.dataset),
                "train_data_batch_size": args.train_batch_size,
                "train_dataloader_num_workers": args.train_num_workers,
                "test_data_batch_size": args.test_batch_size,
                "test_dataloader_num_workers": args.test_num_workers,
                "device": args.device,
                "lr": args.lr,
                "mod_to_train": args.mod,
                "optimizer": args.opt,
                "loss": args.loss,
                "dropout": args.dropout,
                "seed": args.seed or None,
                "mod_init_time_sec": mod_init_time,
                "opt_init_time_sec": opt_init_time
                }
            )

        # define these as step metrics will be different for some metrics
        wandb.define_metric("batch_train")
        wandb.define_metric("batch_test")
        wandb.define_metric("epoch")

        wandb.define_metric("batch_train_loss", step_metric="batch_train", summary='last')
        wandb.define_metric("epoch_train_loss", step_metric="epoch", summary='last')
        wandb.define_metric("batch_test_acc", step_metric="batch_test", summary='last')
        wandb.define_metric("epoch_test_acc", step_metric="epoch", summary='last')        

    # train and eval mod
    train_losses = []
    test_accs = []
    print(f'Starting training using device: {args.device}!')
    for i in range(args.epochs):
        if args.verbose: print(f'Training epoch {i}...')
        train_loss = train(model=mod, criterion=criterion, optimizer=optimizer, train_loader=train_loader, device=args.device, log_wandb=args.log_wandb)
        test_acc = test(model=mod, test_loader=test_loader, device=args.device, log_wandb=args.log_wandb)
        train_losses.append(train_loss)
        test_accs.append(test_acc)
        if args.log_wandb:
            wandb.log({"epoch_train_loss": train_loss, "epoch": i})
            wandb.log({"epoch_test_acc": test_acc, "epoch": i})
    print(f'Done training!')
    print(f'Saving model...')
    torch.save(mod, f'{args.mod_save_path}/{args.mod}.pth')
    print('Done saving model!')
    wandb.finish()

    print()
    print('Summary:')
    print(f'Avg. Train Loss: {sum(train_losses) / args.epochs}')
    print(f'Avg. Test Accuracy: {sum(test_accs) / args.epochs}')