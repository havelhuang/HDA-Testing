from torch import optim
import torch
import time
from torch.autograd import Variable
import torchvision.utils as vutils
from tqdm import tqdm
import utils
# import visual

def generate_reconstructions(data_loader, model):
    model.eval()
    x, _ = data_loader.__iter__().next()
    x = x[:16].cuda()
    x_tilde = model.generate(x)

    x_cat = torch.cat([x, x_tilde], 0)

    vutils.save_image(
        x_cat,
        'samples/reconstructions_data.png',
        normalize=True,
        nrow=4
    )

def train(data_loader, model, optimizer, epoch, cuda):
    model.train()
    data_stream = tqdm(enumerate(data_loader))
    for batch_index, (x, _) in data_stream:
    
            # prepare data on gpu if needed
            x = Variable(x).cuda() if cuda else Variable(x)

            # flush gradients and run the model forward
            optimizer.zero_grad()
            
            result = model(x)
            loss = model.loss_function(*result, M_N = 1)
            reconstruction_loss = loss['Reconstruction_Loss']
            kl_divergence_loss = loss['KLD']
            total_loss = loss['total_loss']

            # backprop gradients from the loss
            total_loss.backward()
            optimizer.step()

            # update progress
            data_stream.set_description((
                'epoch: {epoch} | '
                'progress: [{trained}/{total}] ({progress:.0f}%) | '
                'loss => '
                'total: {total_loss:.7f} / '
                're: {reconstruction_loss:.6f} / '
                'kl: {kl_divergence_loss:.6f}'
            ).format(
                epoch=epoch,
                trained=batch_index * len(x),
                total=len(data_loader.dataset),
                progress=(100. * batch_index / len(data_loader)),
                total_loss=total_loss.data.item(),
                reconstruction_loss=reconstruction_loss.data.item(),
                kl_divergence_loss=kl_divergence_loss.data.item(),
            ))


def test(data_loader, model,cuda):
    model.eval()
    start_time = time.time()
    data_stream = tqdm(enumerate(data_loader))
    with torch.no_grad():
        loss_recons, loss_kld = 0., 0.
        for batch_index, (x, _) in data_stream:
            # prepare data on gpu if needed
            x = Variable(x).cuda() if cuda else Variable(x)
            
            result = model(x)
            loss = model.loss_function(*result, M_N = 1)
            reconstruction_loss = loss['Reconstruction_Loss']
            kl_divergence_loss = loss['KLD']
            total_loss = loss['total_loss']

            loss_recons += reconstruction_loss
            loss_kld += kl_divergence_loss

            data_stream.set_description((
                'progress: [{trained}/{total}] ({progress:.0f}%) | '
                'loss => '
                'total: {total_loss:.7f} / '
                're: {reconstruction_loss:.6f} / '
                'kl: {kl_divergence_loss:.6f}'
            ).format(
                trained=batch_index * len(x),
                total=len(data_loader.dataset),
                progress=(100. * batch_index / len(data_loader)),
                total_loss= total_loss.data.item(),
                reconstruction_loss=reconstruction_loss.data.item(),
                kl_divergence_loss=kl_divergence_loss.data.item(),
            ))


        loss_recons /= len(data_loader)
        loss_kld/= len(data_loader)

    print('\nValidation Completed!\tReconstruction Loss: {:5.4f}\tKLD Loss: {:5.4f} Time: {:5.3f} s'.format(
        loss_recons.item(),
        loss_kld.item(),
        time.time() - start_time
    ))

    return loss_recons.item(), loss_kld.item()

def train_vae_model(model, train_dataset, test_dataset, epochs=10,
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

    # loss_recons, loss_kld = test(test_data_loader, model, cuda)

    BEST_LOSS = 99999
    LAST_SAVED = -1

    for epoch in range(epoch_start, epochs+1):
        
        train(train_data_loader, model, optimizer, epoch, cuda)
        loss_recons, loss_kld = test(test_data_loader, model, cuda)
            
        generate_reconstructions(test_data_loader, model)
        _,images = model.sample(sample_size,cuda) 
        vutils.save_image(images,
                    'samples/sampled_data.png',
                        normalize=True,
                        nrow=4)
                
        print()
        if loss_recons <= BEST_LOSS:
            BEST_LOSS = loss_recons
            LAST_SAVED = epoch
            print("Saving model!")
            utils.save_checkpoint(model, checkpoint_dir, epoch)
        else:
            print("Not saving model! Last saved: {}".format(LAST_SAVED))


