import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from cifar_setter import cifar_10_setter
from hypernet_modules import ResBlock
from primary_network import PrimaryNetwork
from train_funcs import train_model
from utils import *

# set device as cpu if no gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    
    # directory to save model and results
    path = './results'

    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isdir(path):
        raise Exception('%s is not a dir' % path)
        
    # loading data
    dataloaders, dataset_sizes = cifar_10_setter()
    
    # define model
    model = PrimaryNetwork(ResBlock, [9,9,9])
    model = model.to(device)

    # loss criterion
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # Decay LR by a factor of 0.2 every 100 epochs
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100+100*x for x in range(10)], gamma=0.2)

    # train model
    train_loss, valid_loss, train_acc, valid_acc, test_loss, test_acc, best_model_wts = train_model(model, dataloaders, dataset_sizes,
                                                                                                    criterion, optimizer, scheduler, 
                                                                                                    device, num_epochs=1000)
    
    # save model, logs, and graphs
    result_logger(path, train_loss, valid_loss, train_acc, valid_acc, test_loss, test_acc, best_model_wts)
    plotter(path, 'loss', train_loss, valid_loss, test_loss)
    plotter(path, 'acc', train_acc, valid_acc, test_acc)



if __name__ == '__main__':
    main()
    