import time
import numpy as np
import numpy.random as random
import logging
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

from util import T,N
from model import build_models
from datasets import build_bible_datasets
from options import get_options
from eval import eval_sample



class ContrastiveLoss(torch.nn.Module):
    # copied code from: https://gist.github.com/harveyslash/725fcc68df112980328951b3426c0e0b#file-contrastive-loss-py
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # label 0 means same (loss= 1*d^2) 1 means not-same.
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


def train_main(opt,bucket_iter_train=None, bucket_iter_val=None,models=None):
    contrastiveLoss = T(ContrastiveLoss())
    nllloss_for_recon = T(torch.nn.NLLLoss(ignore_index=1)) #ignore padding
    bce = T(torch.nn.BCELoss())
    bce2 = T(torch.nn.BCELoss())


    # --------- training funtions ------------------------------------
    def train(b, models, dont_optimize):


        models.en_sty.zero_grad()
        models.en_sem.zero_grad()
        models.decoder.zero_grad()

        #h_sem0 = models.en_sem(b.sent_0)  # sent0 = b.sent_0 #sem A , style A
        h_sem1 = models.en_sem(b.sent_1)  # sent1 = b.sent_1 #sem A, style B
        h_semX = models.en_sem(b.sent_x)  # sentX = b.sent_x #semAorC , style A
        recon_target = b.sent_0_target  # one-hot

        ######### SIM LOSS #########
        # Original was MSE
        # if you want to use torch criterion, you need to copy the label and set it to not requreing gradiant
        # so below is different than nn.MSELoss()(h_sem0,h_sem1.detach()). I wonder if only one get grad updates!
        # sim_loss = torch.mean(torch.sum(torch.pow(h_sem0- h_sem1,2),dim=1))
        # But constractive loss is more reasnible
        #############################################
        sim_loss = contrastiveLoss(h_sem1, h_semX, T(torch.round(b.is_x_0)))



        ######### RECONSTRUCTION LOSS ######### note: quite slow
        # reconstruct sent0 from semantics of sent1 (==sem of sent0, different style), and style of sent0.
        h_sty0 = models.en_sty(b.sent_0)
        merged = torch.cat([h_sem1, h_sty0], dim=1)
        merged.unsqueeze_(0)  # 32x25 -> 1x32x25 . 1 is for one hidden-layer (not-stacked)
        recon_sent0, _, _ = models.decoder(inputs=recon_target,  # pass not None for teacher focring  (batch, seq_len, input_size)
                                encoder_hidden=merged,  # (num_layers * num_directions, batch_size, hidden_size)
                                encoder_outputs=None,  # pass not None for attention
                                teacher_forcing_ratio=1,
                                function = F.log_softmax
                                # in(0, 1-random.random()* epoch * 0.1) #range 0..1 , must pass inputs if >0. as epochs increase, it's lower
                                )

        # good reconstruction-loss need to:
        # ignore padding (it is easy to guess always <pad> as a result and to be usually right.also called masking)
        # consider length of sentences. is a short 5 word sentnece weight the same as long 40 words sentence?
        #   'elementwise_mean' means look at each word by itself. one can change this to be on sentence level

        # we calculate it manually as sizes may not match in the returned array using seq2seq library
        # in the end , we sum the loss per timestamp and divide by number of timestamps
        acc_loss, norm_term = 0, 0
        for step, timestamp_output in enumerate(recon_sent0):#list of 65 x [32, 2071]
            batch_size = recon_target.size(0)
            if step + 1 >= recon_target.size(1): #in beginning, model might output 200 steps, later will converge to target
                # print ('breaking!!! at step',step)
                break
            gold = recon_target[:, step + 1]  # this is one timstamp across batches
            curr_loss = nllloss_for_recon(timestamp_output, gold)
            acc_loss += curr_loss
            norm_term += 1

        rec_loss = acc_loss / norm_term


        ######### ADV LOSS #########
        h_sty1 = models.en_sty(b.sent_1)
        h_styX = models.en_sty(b.sent_x)
        adv_disc_p = models.adv_disc(torch.cat([h_sty1, h_styX], dim=1))

        adv_target = T(torch.FloatTensor(np.full(shape=(b.sent_0[0].shape[0], 1), fill_value=0.5)))
        # the loss below is a parabula with min at log(0.5)=0.693... see article
        adv_disc_loss = bce(adv_disc_p, adv_target) + np.log(0.5)  # np.log(0.5)=-0.693 ,
        # logger.debug(f'### adv_disc_loss {N(adv_disc_p.data[:5]).T} target={N(adv_target.data[:5]).T} bce={adv_disc_loss}')
        # logger.debug(f'    sanity test: on first step, you expect adv_disc_loss to be near zero')


        ######### BACKWARD #########
        # full loss
        loss = rec_loss + sim_loss * opt.sem_sim_weight + opt.sd_weight * adv_disc_loss  # rec_loss + sim_loss + opt.sd_weight*adv_disc_loss

        if not dont_optimize:
            loss.backward()
            optimizer_en_sem.step()
            optimizer_en_sty.step()
            optimizer_decoder.step()

        return N(sim_loss.data) * opt.sem_sim_weight, (rec_loss.data), N(adv_disc_loss.data) * opt.sd_weight  # N


    def train_scene_discriminator(b,models,dont_optimize):
        models.adv_disc.zero_grad()

        h_sty0    = models.en_sty(T(b.sent_1))
        h_sty0or2 = models.en_sty(T(b.sent_x))  # same style, same or different semantics with random chance
        merged = torch.cat([h_sty0, h_sty0or2], dim=1)
        out = models.adv_disc(merged).flatten()

        y = T(b.is_x_0)
        bce = bce2(out, y)

        if not dont_optimize:
            bce.backward()
            optimizer_adv_disc.step()

        acc = np.round(N(out.detach())) == np.round(N(y))
        logger.debug(f'adv_disc out {out.shape} is_x_0 {y.shape}')
        logger.debug(f'out {out.flatten()} y {y.flatten()} acc {acc} bce {bce.data}')

        acc = acc.reshape(-1)
        acc = acc.sum() / len(acc)
        return N(bce.data), N(acc)


    def one_epoc(epoch,bucket_iter_train,models,dont_optimize):
        logger.debug('one_epoc starts')


        epoch_sim_loss, epoch_rec_loss, epoch_anti_disc_loss, epoch_sd_loss, epoch_sd_acc = 0, 0, 0, 0, 0

        training_batch_generator= None
        for i in range(opt.epoch_size):
            # if i % 10==0 : print ('batch',i,'of',opt.epoch_size)
            logger.debug('next batch')
            b = None
            try:
                b = next(training_batch_generator)
            except:
                logger.debug('creating new iterator')
                training_batch_generator = iter(bucket_iter_train)  # only if no choice... it's 1.5 min
                b = next(training_batch_generator)

            # train scene discriminator
            logger.debug(f'b_sent_0 {b.sent_0[0].shape}{b.sent_0[1].shape}')  # TEXT.reverse(b.sent_0))

            # %time train_scene_discriminator(b) #50ms
            sd_loss, sd_acc = train_scene_discriminator(b,models,dont_optimize)
            logger.debug('train_scene_discriminator done')

            epoch_sd_loss += sd_loss
            epoch_sd_acc += sd_acc

            # train main model
            # %time train(b,epoch) #300ms
            sim_loss, rec_loss, anti_disc_loss = train(b, models, dont_optimize)
            logger.debug('train done')

            epoch_sim_loss += sim_loss
            epoch_rec_loss += rec_loss
            epoch_anti_disc_loss += anti_disc_loss

            logger.setLevel(logging.INFO)
        logger.info('[%02d] %s rec loss: %.4f | sim loss: %.4f | anti_disc_loss: %.4f || scene disc %.4f %.3f%% ' % (
        epoch, "eval" if dont_optimize else "train",epoch_rec_loss / opt.epoch_size,
        epoch_sim_loss / opt.epoch_size, epoch_anti_disc_loss / opt.epoch_size,
        epoch_sd_loss / opt.epoch_size, 100 * epoch_sd_acc / opt.epoch_size))


    def set_train(models):
        models.en_sty.train()  # and not eval() mode
        models.en_sem.train()
        models.decoder.train()
        models.adv_disc.train()


    def set_eval(models):
        models.en_sty.eval()  # and not eval() mode
        models.en_sem.eval()
        models.decoder.eval()
        models.adv_disc.eval()


    # --------- training loop ------------------------------------




    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
    logger.setLevel(logging.INFO)  # not DEBUG

    print ('running train with options:',opt)
    if (bucket_iter_train == None and bucket_iter_val == None and models == None):
        bucket_iter_train, bucket_iter_val = build_bible_datasets(verbose=False)
        models = build_models(bucket_iter_train.dataset, opt)

    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam


    epoc_count = 0
    for lr in [opt.lr, opt.lr / 10, opt.lr / 4]:  # start 0.0001 for 2 big epocs
        print('lr', lr, opt.beta1)
        optimizer_en_sem = optimizer(models.en_sem.parameters(), lr, betas=(opt.beta1, 0.999))
        optimizer_en_sty = optimizer(models.en_sty.parameters(), lr, betas=(opt.beta1, 0.999))
        optimizer_decoder = optimizer(models.decoder.parameters(), lr, betas=(opt.beta1, 0.999))
        optimizer_adv_disc = torch.optim.SGD(models.adv_disc.parameters(), opt.adv_disc_lr)


        bucket_iter_val
        for epoch in range(0, opt.epocs):
            set_train(models)
            one_epoc(epoc_count,bucket_iter_train, models,dont_optimize=False)

            if epoch%10==0:
                set_eval(models)
                one_epoc(epoc_count, bucket_iter_val,  models,dont_optimize=True)
                eval_sample(bucket_iter_val, models)

            epoc_count += 1

    print('training loop done')

    return bucket_iter_train, bucket_iter_val,models


#train_main(opt = get_options(False))