import torch
import time
import os
import copy


def phase_epoch(phase, model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device):
    
    """
    Epoch training function. Updates model at training time and test model at valid time.
    
    phase : phase of epoch ex) 'train', 'valid', 'test'
    model : model to be trained
    dataloaders : dataloaders dictionary
    """

    if phase == 'train':
        scheduler.step()
        model.train()  # Set model to training mode
    else:
        model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        phase, epoch_loss, epoch_acc))
    
    return epoch_loss, epoch_acc


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=250):
    
    """
    Main training function.
    """
    
    since = time.time()
    
    train_loss, valid_loss, train_acc, valid_acc = [], [], [], []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                train_epoch_loss, train_epoch_acc = phase_epoch(phase, model, dataloaders, dataset_sizes,
                                                                criterion, optimizer, scheduler, device)
                train_loss.append(train_epoch_loss)
                train_acc.append(train_epoch_acc)
                
            else:
                valid_epoch_loss, valid_epoch_acc = phase_epoch(phase, model, dataloaders, dataset_sizes,
                                                                criterion, optimizer, scheduler, device)
                
                valid_loss.append(valid_epoch_loss)
                valid_acc.append(valid_epoch_acc)
                
                # deep copy the model
                if valid_epoch_acc > best_acc:
                        best_acc = valid_epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                print()               
        
        # estimate training time
        if epoch == 0:
            time_estimator = (time.time() - since) * num_epochs
            print('Estimated training time is {:.0f}h {:.0f}m'.format(time_estimator // 3600, (time_estimator % 3600) // 60))
    
    else:
        phase = 'test'
        test_loss, test_acc = phase_epoch(phase, model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device)
        print()

    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return train_loss, valid_loss, train_acc, valid_acc, test_loss, test_acc, best_model_wts
