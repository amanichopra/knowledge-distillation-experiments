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

def soft_target_loss(soft_targets, soft_prob, temperature):
    return -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (temperature**2)

# function to train model for 1 epoch using cross-entropy and soft-target loss
def distill_knowledge_ce_stl(teacher, student, optimizer, train_loader, device, log_wandb, temperature, ce_weight, st_weight, profile_path):
    student.train()
    teacher.eval()

    running_loss = 0.0

    if profile_path:
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_path),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
                with torch.no_grad():
                    teacher_logits, _ = teacher(inputs)

                # Forward pass with the student model
                student_logits, _ = student(inputs)

                #Soften the student logits by applying softmax first and log() second
                soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=-1)
                soft_prob = nn.functional.log_softmax(student_logits / temperature, dim=-1)

                # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
                soft_targets_loss = soft_target_loss(soft_targets, soft_prob, temperature)

                # Calculate the true label loss
                ce_loss = nn.CrossEntropyLoss()
                label_loss = ce_loss(student_logits, labels)

                # Weighted sum of the two losses
                loss = st_weight * soft_targets_loss + ce_weight * label_loss

                loss.backward()
                optimizer.step()
            
                if log_wandb:
                    wandb.log({"batch_train_loss": loss.item(), "batch_train": train_batch_counter})
                    distill_knowledge_ce_stl.train_batch_counter += 1
                running_loss += loss.item()
            prof.step()
        prof.export_chrome_trace(profile_path)


    else:
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits, _ = teacher(inputs)

            # Forward pass with the student model
            student_logits, _ = student(inputs)

            #Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / temperature, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = soft_target_loss(soft_targets, soft_prob, temperature)

            # Calculate the true label loss
            ce_loss = nn.CrossEntropyLoss()
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = st_weight * soft_targets_loss + ce_weight * label_loss

            loss.backward()
            optimizer.step()
            if log_wandb:
                wandb.log({"batch_train_loss": loss.item(), "batch_train": train_batch_counter})
                distill_knowledge_ce_stl.train_batch_counter += 1
            running_loss += loss.item()

    return running_loss / len(train_loader)

def distill_knowledge_ce_csl(teacher, student, optimizer, train_loader, device, log_wandb, ce_weight, cs_weight, profile_path):
    student.train()
    teacher.eval()

    running_loss = 0.0
    if profile_path:
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_path),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
                with torch.no_grad():
                    _, teacher_feats = teacher(inputs)

                # Forward pass with the student model
                student_logits, student_feats = student(inputs)

                # Calculate the cosine loss. Target is a vector of ones. From the loss formula above we can see that is the case where loss minimization leads to cosine similarity increase.
                cosine_loss = nn.CosineEmbeddingLoss()
                feats_loss = cosine_loss(student_feats, teacher_feats, target=torch.ones(inputs.size(0)).to(device))

                # Calculate the true label loss
                ce_loss = nn.CrossEntropyLoss()
                label_loss = ce_loss(student_logits, labels)

                # Weighted sum of the two losses
                loss = cs_weight * feats_loss + ce_weight * label_loss

                loss.backward()
                optimizer.step()
                if log_wandb:
                    wandb.log({"batch_train_loss": loss.item(), "batch_train": distill_knowledge_ce_csl.train_batch_counter})
                    distill_knowledge_ce_csl.train_batch_counter += 1
                running_loss += loss.item()
            
            prof.step()
        prof.export_chrome_trace(profile_path)
    else:
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                _, teacher_feats = teacher(inputs)

            # Forward pass with the student model
            student_logits, student_feats = student(inputs)

            # Calculate the cosine loss. Target is a vector of ones. From the loss formula above we can see that is the case where loss minimization leads to cosine similarity increase.
            cosine_loss = nn.CosineEmbeddingLoss()
            feats_loss = cosine_loss(student_feats, teacher_feats, target=torch.ones(inputs.size(0)).to(device))

            # Calculate the true label loss
            ce_loss = nn.CrossEntropyLoss()
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = cs_weight * feats_loss + ce_weight * label_loss

            loss.backward()
            optimizer.step()
            if log_wandb:
                wandb.log({"batch_train_loss": loss.item(), "batch_train": distill_knowledge_ce_csl.train_batch_counter})
                distill_knowledge_ce_csl.train_batch_counter += 1
            running_loss += loss.item()


    return running_loss / len(train_loader)

def distill_knowledge_ce_mse(teacher, student, optimizer, train_loader, device, log_wandb, ce_weight, mse_weight, profile_path):
    student.train()
    teacher.eval()

    running_loss = 0.0
    if profile_path:
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_path),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
                with torch.no_grad():
                    _, teacher_feats = teacher(inputs)

                # Forward pass with the student model
                student_logits, student_feats = student(inputs)

                # Calculate the MSE loss. 
                mse_loss = nn.MSELoss()
                feats_loss = mse_loss(student_feats, teacher_feats)

                # Calculate the true label loss
                ce_loss = nn.CrossEntropyLoss()
                label_loss = ce_loss(student_logits, labels)

                # Weighted sum of the two losses
                loss = mse_weight * feats_loss + ce_weight * label_loss

                loss.backward()
                optimizer.step()
                if log_wandb:
                    wandb.log({"batch_train_loss": loss.item(), "batch_train": distill_knowledge_ce_mse.train_batch_counter})
                    distill_knowledge_ce_mse.train_batch_counter += 1
                running_loss += loss.item()
            prof.step()
        prof.export_chrome_trace(profile_path)
    else:

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                _, teacher_feats = teacher(inputs)

            # Forward pass with the student model
            student_logits, student_feats = student(inputs)

            # Calculate the MSE loss. 
            mse_loss = nn.MSELoss()
            feats_loss = mse_loss(student_feats, teacher_feats)

            # Calculate the true label loss
            ce_loss = nn.CrossEntropyLoss()
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = mse_weight * feats_loss + ce_weight * label_loss

            loss.backward()
            optimizer.step()
            if log_wandb:
                wandb.log({"batch_train_loss": loss.item(), "batch_train": train_batch_counter})
                distill_knowledge_ce_mse.train_batch_counter += 1
            running_loss += loss.item()

    return running_loss / len(train_loader)

# function to evaluate model for 1 epoch
def test(model, test_loader, device, log_wandb):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            batch_total = labels.size(0)
            batch_correct = (predicted == labels).sum().item()
            if log_wandb:
                wandb.log({"batch_test_acc": 100 * batch_correct / batch_total, "batch_test": test_batch_counter})
                test.test_batch_counter += 1
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
    parser.add_argument('--loss', help='Type of loss function to use during training.', default='ce-stl', choices=['ce-stl', 'ce-csl', 'ce-mse'])
    parser.add_argument('--opt', help='Type of optimizer to use during training.', default='adam', choices=['adam', 'sgd', 'adagrad', 'adadelta', 'rmsprop'])
    parser.add_argument('--seed', help='PyTorch seed to use for reproducability.', default=None, type=int)
    parser.add_argument('--student_mod', help='Name of model to train.', choices=['cnn', 'resnet18', 'resnet18_pt'])
    parser.add_argument('--teacher_mod', help='Name of model to distill knowledge from.', choices=['cnn', 'cnn_pt', 'resnet34', 'resnet50', 'resnet34_pt', 'resnet50_pt'])
    parser.add_argument('--verbose', help='Whether or not to print debugging outputs.', type=int, default=1)
    parser.add_argument('--log_wandb', help='Whether or not to log metrics to Wandb.', type=int, default=0)
    parser.add_argument('--ce_w', help='Weight to use for cross-entropy component of KD loss.', type=float, default=0.75)
    parser.add_argument('--st_w', help='Weight to use for soft-target component of KD loss.', type=float, default=0.25)
    parser.add_argument('--cs_w', help='Weight to use for cosine similarity component of KD loss.', type=float, default=0.25)
    parser.add_argument('--mse_w', help='Weight to use for MSE component of KD loss.', type=float, default=0.25)
    parser.add_argument('--temp', help='Value to use for temperature when computing KD loss.', type=float, default=2)
    parser.add_argument('--teacher_mod_path', help='Path to load pretrained teacher model.', default=None)
    parser.add_argument('--profile', help='Whether or not to profile code.', type=int, default=0)

    args = parser.parse_args()

    distill_knowledge_ce_csl.train_batch_counter = 0
    distill_knowledge_ce_stl.train_batch_counter = 0
    distill_knowledge_ce_mse.train_batch_counter = 0
    test.test_batch_counter = 0

    if args.seed:
        torch.manual_seed(args.seed)
    
    profile_path = None
    if args.profile:
        profile_path = f'./log/{args.student_mod}_student-{args.teacher_mod}_teacher.json'
    
    # load data
    if args.verbose: print('Loading data...')
    if 'resnet' in args.student_mod and 'resnet' in args.teacher_mod:
        img_size = 224
    elif 'cnn' in args.student_mod and 'cnn' in args.teacher_mod:
        img_size = 32
    train_data, test_data = get_cifar10_datasets(img_size=img_size)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, num_workers=args.train_num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.test_num_workers)
    if args.verbose:
        print(f'Number of training examples: {len(train_loader)}')
        print(f'Number of testing examples: {len(test_loader)}')
        print('Done loading data!')

    # initialize student and teacher models
    if args.verbose: print('Initializing model...')
    start = perf_counter()
    
    if args.student_mod == 'cnn':
        student = CNNStudent(num_classes=10, dropout=args.dropout)
    elif args.student_mod == 'resnet18':
        student = Resnet18(pretrained=False)
    elif args.student_mod == 'resnet18_pt':
        student = Resnet18(pretrained=True)

    if args.teacher_mod_path:
        teacher = torch.load(args.teacher_mod_path)
    elif args.teacher_mod == 'cnn':
        teacher = CNNTeacher(num_classes=10, dropout=args.dropout)
    elif args.teacher_mod == 'resnet34':
        teacher = Resnet34(pretrained=False)
    elif args.teacher_mod == 'resnet34_pt':
        teacher = Resnet34(pretrained=True)
    elif args.teacher_mod == 'resnet50':
        teacher = Resnet50(pretrained=False)
    elif args.teacher_mod == 'resnet50_pt':
        teacher = Resnet50(pretrained=True)

    student = student.to(args.device)
    teacher = teacher.to(args.device)
    end = perf_counter()
    mod_init_time = end - start
    if args.verbose: print('Done initializing model!')

    if args.verbose: print('Initializing model...')
    # initialize optimizer
    start = perf_counter()
    if args.opt == 'adam':
        optimizer = optim.Adam(student.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(student.parameters(), lr=args.lr)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(student.parameters(), lr=args.lr)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(student.parameters(), lr=args.lr)
    elif args.opt == 'adadelta':
        optimizer = optim.Adadelta(student.parameters(), lr=args.lr)
    end = perf_counter()
    opt_init_time = end - start
    
    # init wandb to log training metrics
    if args.log_wandb:
        wandb.init(
                # set the wandb project where this run will be logged
                project="knowledge-distillation-experiments",
                
                # track hyperparameters and run metadata
                config={
                "exp_name": 'kd_training',
                "epochs": args.epochs,
                "train_data_size": len(train_loader.dataset),
                "test_data_size": len(test_loader.dataset),
                "train_data_batch_size": args.train_batch_size,
                "train_dataloader_num_workers": args.train_num_workers,
                "test_data_batch_size": args.test_batch_size,
                "test_dataloader_num_workers": args.test_num_workers,
                "device": args.device,
                "lr": args.lr,
                "student_mod_to_train": args.student_mod,
                "teacher_mod_to_train": args.teacher_mod,
                "optimizer": args.opt,
                "loss": args.loss,
                "dropout": args.dropout,
                "seed": args.seed or None,
                "mod_init_time_sec": mod_init_time,
                "opt_init_time_sec": opt_init_time
                }
            )

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
    if args.verbose: print(f'Starting training using device: {args.device}!')
    for i in range(args.epochs):
        if args.verbose: print(f'Training epoch {i}...')
        if args.loss == 'ce-stl':
            train_loss = distill_knowledge_ce_stl(student=student, teacher=teacher, optimizer=optimizer, train_loader=train_loader, device=args.device, log_wandb=args.log_wandb, ce_weight=args.ce_w, st_weight=args.st_w, temperature=args.temp, profile_path=profile_path)
        elif args.loss == 'ce-csl':
            train_loss = distill_knowledge_ce_csl(teacher=teacher, student=student, optimizer=optimizer, train_loader=train_loader, device=args.device, log_wandb=args.log_wandb, ce_weight=args.ce_w, cs_weight=args.cs_w, profile_path=profile_path)
        elif args.loss == 'ce-mse':
            train_loss = distill_knowledge_ce_mse(teacher=teacher, student=student, optimizer=optimizer, train_loader=train_loader, device=args.device, log_wandb=args.log_wandb, ce_weight=args.ce_w, mse_weight=args.mse_w, profile_path=profile_path)
        test_acc = test(model=student, test_loader=test_loader, device=args.device, log_wandb=args.log_wandb)
        train_losses.append(train_loss)
        test_accs.append(test_acc)
        if args.log_wandb:
            wandb.log({"epoch_train_loss": train_loss, "epoch": i})
            wandb.log({"epoch_test_acc": test_acc, "epoch": i})

    
    wandb.finish()
    if args.verbose:
        print(f'Done training!')
        print()
        print('Summary:')
        print(f'Avg. Train Loss: {sum(train_losses) / args.epochs}')
        print(f'Avg. Test Accuracy: {sum(test_accs) / args.epochs}')