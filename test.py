import utils
import torch
import os
import torch.nn as nn
from model.inception import InceptionV3
from GaussianKDE import GaussianKDE
from sklearn.neighbors import KernelDensity
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.categorical import Categorical
import torchvision.utils as vutils
import numpy as np
from utils import get_nearest_oppo_dist, cal_gradient, fitness_score, mutation, cal_robust
import numpy as np
from utils import calculate_fid
import time

def two_step_ga(model, x_seed, y_seed, eps, local_op, n_particles = 300, n_mate = 20, max_itr = 50, alpha = 1.00):
    adv_images = x_seed.repeat(n_particles,1,1,1)
    delta = torch.empty_like(adv_images).normal_(mean=0.0,std=0.01)
    delta = torch.clamp(delta, min=-eps, max=eps)
    adv_images = torch.clamp(x_seed + delta, min=0, max=1).detach()

    for i in range(max_itr):
        obj, loss, op_loss = fitness_score(x_seed, y_seed, adv_images, model, local_op,alpha)
        sorted, indices = torch.sort(obj, dim=-1, descending=True)
        parents = adv_images[indices[:n_mate]]
        obj_parents = sorted[:n_mate]

        # Generating next generation using crossover
        m = Categorical(logits=obj_parents)
        parents_list = m.sample(torch.Size([2*n_particles]))
        parents1 = parents[parents_list[:n_particles]]
        parents2 = parents[parents_list[n_particles:]]
        pp1 = obj_parents[parents_list[:n_particles]]
        pp2 = obj_parents[parents_list[n_particles:]]
        pp2 = pp2 / (pp1+pp2)
        pp2 = pp2[(..., ) + (None,)*3]

        mask_a = torch.empty_like(parents1).uniform_() > pp2
        mask_b = ~mask_a
        parents1[mask_a] = 0.0
        parents2[mask_b] = 0.0
        children = parents1 + parents2

        # add some mutations to children and genrate test set for next generation
        children = mutation(x_seed, children, eps, p=0.2)
        adv_images = torch.cat([children,parents], dim=0)

    obj, loss, op_loss = fitness_score(x_seed, y_seed, adv_images, model, local_op,alpha)
    sorted, indices = torch.sort(loss, dim=-1, descending=True)
    return adv_images[indices[:10]], loss[indices[:10]], op_loss[indices[:10]]


def test_model(vae, model, dataset, batch_size, latent_dim, data_config, output_dir, eps, n_seeds, local_op, cuda):

    torch.manual_seed(0)
    vae.eval()
    model.eval()


    n_channel  = data_config['channels']
    img_size = data_config['size']
    n_class = data_config['classes']

    n = len(dataset)
    data_loader = utils.get_data_loader(dataset, batch_size, cuda = cuda)

    grad_norm = []
    x_act_dist = []

    # Get data into arrays for convenience
    if cuda:
        x_test = torch.zeros(n, n_channel, img_size, img_size).cuda()
        y_test = torch.zeros(n, dtype = int).cuda()
        y_pred = torch.zeros(n, dtype = int).cuda()
        x_mu = torch.zeros(n, latent_dim).cuda()
        x_std = torch.zeros(n, latent_dim).cuda()

    else:
        x_test = torch.zeros(n, n_channel, img_size, img_size)
        y_test = torch.zeros(n, dtype = int)
        x_mu = torch.zeros(n, latent_dim)
        x_std = torch.zeros(n, latent_dim)


    
    for idx, (data, target) in enumerate(data_loader):
        if cuda:
            data, target = data.float().cuda(), target.long().cuda()
        else:
            data, target = data.float(), target.long()

        if len(target.size()) > 1:
            target = torch.argmax(target, dim=1)

        grad_batch = cal_gradient(model,data,target)
        grad_norm.append(grad_batch)

        with torch.no_grad():
            mu, log_var = vae.encode(data)
            hidden_act = model.hidden_act(data)
            target_pred = torch.argmax(model(data/2+0.5), dim=1)

            x_test[(idx * batch_size):((idx + 1) * batch_size), :, :, :] = data
            y_test[(idx * batch_size):((idx + 1) * batch_size)] = target
            y_pred[(idx * batch_size):((idx + 1) * batch_size)] = target_pred
            x_mu[(idx * batch_size):((idx + 1) * batch_size), :] = mu
            x_std[(idx * batch_size):((idx + 1) * batch_size), :] = torch.exp(0.5 * log_var)
            x_act_dist.append(hidden_act)

    grad_norm = torch.cat(grad_norm, dim=0)
    x_act_dist = torch.cat(x_act_dist, dim=0)

    indices = torch.where(y_pred==y_test)
    x_test = x_test[indices] 
    y_test = y_test[indices]
    x_mu = x_mu[indices]
    x_std = x_std[indices]
    grad_norm = grad_norm[indices]
    x_act_dist = x_act_dist[indices]

    indices = torch.randperm(len(x_test))[:10000]
    x_test = x_test[indices] 
    y_test = y_test[indices]
    x_mu = x_mu[indices]
    x_std = x_std[indices]
    grad_norm = grad_norm[indices]
    x_act_dist = x_act_dist[indices]



    # # # ################################################################################################### 
    start = time.time()     
    print()
    print('Start to test the model!')
    print()
    print('Dataset:', model.label)
    print('No. of test seeds:', n_seeds)
    print('Total No. in test set:', len(x_mu))
    print('Norm ball radius:', eps)


    kde = GaussianKDE(x_mu, x_std)
    pd = kde.score_samples(x_mu)
  
    grad_aux = utils.min_max_scale(grad_norm.cpu())

    aux_inf = pd * grad_aux
    sorted, indices = torch.sort(aux_inf, dim=-1, descending=True)

    x_seeds = x_test[indices[:n_seeds]]
    y_seeds = y_test[indices[:n_seeds]]
    op = pd[indices[:n_seeds]]

    # compare with random seeds density
    rand_indices = torch.randperm(len(pd))[:n_seeds]
    op_rand = pd[rand_indices]
    x_seeds_rand = x_test[rand_indices] 
    y_seeds_rand = y_test[rand_indices]

    print('################# Global Seeds Probability Density (normalized) ###########################')
    print('KDE probability density:', (sum(op)/sum(pd)).item())
    print('Rand. probability density:', (sum(op_rand)/sum(pd)).item())

    # torch.save([x_seeds,y_seeds,op],'test_seeds/op_'+ model.label +'_seed.pt')
    # torch.save([x_seeds_rand,y_seeds_rand,op_rand],'test_seeds/rand_'+ model.label +'_seed.pt')
    ######################################################################################################
    # # # correlation between auxillary information and robustness
    # # np.save('grad.npy', np.array(grad_norm.cpu()))

    # nns, ret = get_nearest_oppo_dist(np.array(x_act_dist.cpu()), np.array(y_test.cpu()), np.inf, n_jobs=10)
    # # np.save('act_dist.npy', np.array(ret))


    # # print('the minimum class separation is ', min(ret))
    # robustness = []

    # for i in range(10):
    #     lg_p = cal_robust(x_seeds[i], y_seeds[i], model, cuda, grey_scale = True)
    #     print('------------------------------')
    #     robustness.append(lg_p)
        
    # print('avg pmi', np.mean(robustness))
    # print('------------------------------')

    # # np.save('robust.npy', np.array(robustness))
    # # ###################################################################################################
    # test case generation with Two-step Genetic Algorithm
    # GA settings
   
    n_particles = 1000
    n_mate = 20
    max_itr = 100
    alpha = 1.00 

    save_dir = output_dir+'/HDA_'+ local_op
    save_seeds_dir = save_dir+'/test_seeds'
    save_aes_dir = save_dir+'/AEs'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_seeds_dir)
        os.makedirs(save_aes_dir)

    count = 0
    adv_count = 0
    test_set = []
    adv_set = []
    test_label = []
    loss_set = []
    seed_set = []

    with torch.no_grad():
        for x_seed, y_seed in zip(x_seeds,y_seeds):
            x_seed = x_seed /2 + 0.5

            # save test seeds to files
            vutils.save_image(
            x_seed,
            save_seeds_dir+'/{no}_{label}.png'.format(no = count, label=y_seed.item()),
            normalize=False)

            # start running GA on seeds input
            torch.cuda.empty_cache()
            ae, loss, op_loss = two_step_ga(model, x_seed, y_seed, eps, local_op, n_particles = n_particles, n_mate = n_mate, max_itr = max_itr, alpha = alpha)
            
            test_set.append(ae.cpu())
            test_label.append(torch.stack(10*[y_seed]).cpu())

            idx = torch.where(loss>=0)[0]
            if len(idx)>0:
                ae = ae[idx[0]]
                ae_loss = loss[idx[0]]
                ae_pred = torch.argmax(model(ae.unsqueeze(0)), dim=1)

                adv_set.append(ae.unsqueeze(0))
                seed_set.append(x_seed.unsqueeze(0))
                loss_set.append(ae_loss.cpu().unsqueeze(0))
                adv_count += 1

                vutils.save_image(
                    ae,
                    save_aes_dir+'/{no}_{label}_{pred}.png'.format(no = count, label=y_seed.item(), pred = ae_pred.item()),
                    normalize=False)



            count += 1


    # torch.save([test_set,test_label],'AEs_fine_tune_10/'+ model.label +'_hda.pt')


    # calculate l_inf norm beween seeds and AEs
    adv_set = torch.cat(adv_set)
    seed_set = torch.cat(seed_set)
    loss_set = torch.cat(loss_set)
    print('################# Local AEs Perceptual Quality ###########################')
    epsilon = torch.norm(adv_set-seed_set,p=float('inf'),dim=(1,2,3))
    print('Avg. Perturbation Amount:',torch.mean(epsilon).item())

    # calculate FID between seeds and AEs
    if model.label == 'mnist' or model.label == 'FashionMnist':
        with torch.no_grad():
            adv_mu, adv_var = vae.encode((adv_set-0.5)*2)
            seed_mu, seed_var = vae.encode((seed_set-0.5)*2)
        fid = calculate_fid(np.array(adv_mu.cpu()),np.array(seed_mu.cpu()))
    else:
        # prepare for calculating fid
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        inception = InceptionV3([block_idx]).cuda()

        with torch.no_grad():
            adv_mu = inception(adv_set)[0]
            seed_mu = inception(seed_set)[0]
            adv_mu= torch.flatten(adv_mu, start_dim = 1)
            seed_mu = torch.flatten(seed_mu, start_dim = 1)
        fid = calculate_fid(np.array(adv_mu.cpu()),np.array(seed_mu.cpu()))

    end = time.time()

    print('FID (adv.): %.3f' % fid)
    print('Attack success rate:', adv_count/n_seeds)
    print('Avg. prediction loss of AEs:', torch.mean(loss_set).item())
    print('Elapsed time:',end - start)

   
    # generate a test report
    f = open(save_dir+ "/test_report.txt", "w")
    f.write('Dataset: {}\n'.format(model.label))
    f.write('Total No. of test seeds: {}\n'.format(n_seeds))
    f.write('Norm ball radius: {}\n'.format(eps))
    f.write('################# Global Seeds Probability Density ###########################\n')
    f.write('KDE probability density: {}\n'.format((sum(op)/sum(pd)).item()))
    f.write('Rand. probability density: {}\n'.format((sum(op_rand)/sum(pd)).item()))
    f.write('################# Local AEs Perceptual Quality ###########################\n')
    f.write('Avg. Perturbation Amount: {}\n'.format(torch.mean(epsilon).item()))
    f.write('FID (adv.): {}\n'.format(fid))
    f.write('Detect AEs success rate: {}\n'.format(adv_count/n_seeds))
    f.write('Avg. prediction loss of AEs: {}\n'.format(torch.mean(loss_set).item()))
    f.write('Elapsed time: {}\n'.format(end - start))
    f.close()

    



    