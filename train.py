from torch import optim
import torch
import torch.nn as nn
import time
from torch.autograd import Variable
import torchvision.utils as vutils
from tqdm import tqdm
import utils

def train(data_loader, model, optimizer, epoch, cuda):
    model.train()
    running_loss = 0.0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    data_stream = tqdm(enumerate(data_loader))
    for batch_index, (x, y) in data_stream:

        # prepare data on gpu if needed
        if cuda:
            x = x.to('cuda')
            y = y.to('cuda')

        x = x/2 + 0.5
        
        if len(y.size()) > 1:
            real_labels = torch.argmax(y, dim=1)
        else:
            real_labels = y
        
        # flush gradients and run the model forward
        optimizer.zero_grad()
        
        result = model(x)
        loss = criterion(result, real_labels)

        # backprop gradients from the loss
        loss.backward()
        optimizer.step()

        pred_labels = torch.argmax(result, dim=1)
        correct += (pred_labels == real_labels).sum().item() 
        running_loss += loss.item()

        # update progress
        data_stream.set_description((
            'epoch: {epoch} | '
            'progress: [{trained}/{total}] ({progress:.0f}%) | '
            ' => '
            'loss: {loss:.7f} / '
        ).format(
            epoch=epoch,
            trained=batch_index * len(x),
            total=len(data_loader.dataset),
            progress=(100. * batch_index / len(data_loader)),
            loss = loss.data.item()
        ))
    
    acc = correct / len(data_loader.dataset)
    running_loss /= len(data_loader)
    print('\nTraining set: Epoch: %d, Loss: %.5f, Accuracy: %.2f %%' % (epoch, running_loss, 100. * acc))


def test(data_loader, model,cuda):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    data_stream = tqdm(enumerate(data_loader))
    with torch.no_grad():
        running_loss = 0.0
        correct = 0
        for batch_index, (x, y) in data_stream:
            # prepare data on gpu if needed
            if cuda:
                x = x.to('cuda')
                y = y.to('cuda')

            x = x/2 + 0.5

            if len(y.size()) > 1:
                real_labels = torch.argmax(y, dim=1)
            else:
                real_labels = y

            result = model(x)
            loss = criterion(result, real_labels)
            running_loss += loss.item()

            pred_labels = torch.argmax(result, dim=1)
            correct += (pred_labels == real_labels).sum().item()

            data_stream.set_description((
                'progress: [{trained}/{total}] ({progress:.0f}%) | '
                ' => '
                'loss: {total_loss:.7f} / '
            ).format(
                trained=batch_index * len(x),
                total=len(data_loader.dataset),
                progress=(100. * batch_index / len(data_loader)),
                total_loss= loss,
            ))

        acc = correct / len(data_loader.dataset)
        running_loss /= len(data_loader)

    print('\nTest set: Loss: %.5f, Accuracy: %.2f %%' % (running_loss, 100. * acc))

    return acc


def train_model(model, train_dataset, test_dataset, epochs=10,
                batch_size=32, sample_size=32,
                lr=3e-04, weight_decay=1e-5,
                checkpoint_dir='./checkpoints',
                resume=False,
                cuda=False):

    # prepare optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=weight_decay,
    )

    if resume:
        epoch_start = utils.load_checkpoint(model, checkpoint_dir)
    else:
        epoch_start = 1

    train_data_loader = utils.get_data_loader(train_dataset, batch_size, cuda=cuda)
    test_data_loader = utils.get_data_loader(test_dataset, batch_size, cuda=cuda)

    BEST_acc = 0.0
    LAST_SAVED = -1

    for epoch in range(epoch_start, epochs+1):
        
        train(train_data_loader, model, optimizer, epoch, cuda)
        acc = test(test_data_loader, model, cuda)
            
        print()
        if acc >= BEST_acc:
            BEST_acc = acc
            LAST_SAVED = epoch
            print("Saving model!")
            utils.save_checkpoint(model, checkpoint_dir, epoch)
        else:
            print("Not saving model! Last saved: {}".format(LAST_SAVED))

