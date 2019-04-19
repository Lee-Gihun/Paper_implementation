import torch
import os
import matplotlib.pyplot as plt

def result_logger(path, train_loss, valid_loss, train_acc, valid_acc, test_loss, test_acc, best_model_wts):
    
    """
    Saves model weights with .pth file and logs with .csv file
    """
    
    torch.save(best_model_wts, os.path.join(path, 'model.dat'))
      
    with open(os.path.join(path, 'results.csv'), 'w') as f:
        f.write('epoch,train_loss,train_acc,valid_loss,valid_acc,test\n')
        for i in range(len(train_loss)):
            f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,\n' % ((i + 1), train_loss[i], 
                                                         train_acc[i], valid_loss[i], valid_acc[i],))
        else:
            f.write(',,,,,%0.5f\n' % (test_loss))
            f.write(',,,,,%0.5f\n' % (test_acc))
            print('model and logs are saved')
            
            
def plotter(path, mode, train_elem, valid_elem, test_elem):
    
    """
    plots loss or accuracy graph for train/valid logs, and saves as .png file.
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.plot([x+1 for x in range(len(train_elem))], train_elem, 'r+', linestyle='dashed', label='train_%s'%mode)
    plt.plot([x+1 for x in range(len(valid_elem))], valid_elem, 'b+', linestyle='dashed', label='valid_%s'%mode)
    plt.plot(len(train_elem), test_elem, 'c+', label='test_{}({:0.4f})'.format(mode, test_elem))
    plt.xlabel('Epoch')
    plt.ylabel(mode)
    plt.legend()
    plt.grid()
    plt.xticks([x+1 for x in range(len(train_elem))])
    #ax.set_yscale('log')
    fname = os.path.join(path, mode+'.png')
    plt.savefig(fname)
    print('%s plot saved'%mode)
    