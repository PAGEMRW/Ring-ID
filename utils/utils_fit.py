import os

import torch
import torch.nn as nn
from tqdm import tqdm
from .train_utils import get_lr


def fit_one_epoch(model, loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, genval, Epoch, cuda, save_period, save_dir):
    total_loss      = 0
    total_accuracy  = 0

    val_loss            = 0
    val_total_accuracy  = 0
    

    print('Start Train')
    pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images  = images.cuda()
                targets = targets.cuda()

        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        outputs = model(images)
        output  = loss(outputs, targets)

        output.backward()
        optimizer.step()

        with torch.no_grad():
            equal       = torch.eq(torch.round(nn.Sigmoid()(outputs)), targets)
            accuracy    = torch.mean(equal.float())

        total_loss      += output.item()
        total_accuracy  += accuracy.item()

        pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                            'acc'       : total_accuracy / (iteration + 1),
                            'lr'        : get_lr(optimizer)})
        pbar.update(1)
    pbar.close()
    print('Finish Train')
    print('Start Validation')
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model.eval()
    for iteration, batch in enumerate(genval):
        if iteration >= epoch_step_val:
            break
        
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images  = images.cuda()
                targets = targets.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            output  = loss(outputs, targets)

            equal       = torch.eq(torch.round(nn.Sigmoid()(outputs)), targets)
            accuracy    = torch.mean(equal.float())

        val_loss            += output.item()
        val_total_accuracy  += accuracy.item()

        pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1), 
                                'acc'       : val_total_accuracy / (iteration + 1)})
        pbar.update(1)
                
    pbar.close()
    print('Finish Validation')
    loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    
    #-----------------------------------------------#
    #   保存权值
    #-----------------------------------------------#
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

    if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
        
    torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))