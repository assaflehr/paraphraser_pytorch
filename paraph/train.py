import time
import numpy as np
import logging
import time
import torch
import torch.nn.functional as F
from util import T,N
from model import build_models
from datasets import build_bible_datasets
from options import get_options
import numpy.random as random




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


def train_main(opt):
    contrastiveLoss = T(ContrastiveLoss())
    nll = T(torch.nn.NLLLoss())
    bce = T(torch.nn.BCELoss())
    bce2 = T(torch.nn.BCELoss())


    # --------- training funtions ------------------------------------
    def train(b, models, dont_optimize=False):
        # x[0] semantic0   , style0
        # x[1] semantic0   , style1
        # x[2] semantic0orX, style0 (1/2 the time 0 , half X)
        merge_dim=1
        models.en_sty.zero_grad()
        models.en_sem.zero_grad()
        models.decoder.zero_grad()

        # sent0 = b.sent_0 #sem A , style A
        # sent1 = b.sent_1 #sem A, style B
        # sentX = b.sent_x #semAorC , style A
        recon_target = b.sent_0_target  # one-hot

        # logger.debug(f'sent_0 {b.sent_0[0].shape} {b.sent_0[1].shape}')

        h_sem0 = models.en_sem(b.sent_0)
        h_sem1 = models.en_sem(b.sent_1)
        h_semX = models.en_sem(b.sent_x)

        ######### SIM LOSS #########
        # Original was MSE
        # if you want to use torch criterion, you need to copy the label and set it to not requreing gradiant
        # so below is different than nn.MSELoss()(h_sem0,h_sem1.detach()). I wonder if only one get grad updates!
        # sim_loss = torch.mean(torch.sum(torch.pow(h_sem0- h_sem1,2),dim=1))
        #############################################
        # But constractive loss is more reasnible
        # sim_loss = ContrastiveLoss()(h_sem1,h_semX,T(torch.round(b.is_x_0)))
        sim_loss = contrastiveLoss(h_sem1, h_semX, T(torch.round(b.is_x_0)))
        # logger.debug(f'sem_loss: {h_sem1.shape} {h_semX.shape} {sim_loss.shape} {sim_loss} {T(torch.round(b.is_x_0))}')
        # TODO: is it 0 or 1???????????????
        # for k in range(3):
        #  logger.debug(f'###########sim_loss: CURRENTLY USING sem1 . sem1 == more sim loss')


        ######### RECONSTRUCTION LOSS #########
        # reconstruct sent0 from semantics of sent1 (==sem of sent0, different style), and style of sent0.
        h_sty0 = models.en_sty(b.sent_0)
        # This is slow: 25 seconds(!!!!) for the RECONSTRCUTION LOSS CALC. ALL OTHER 15s
        merged = torch.cat([h_sem1, h_sty0], dim=merge_dim)
        merged.unsqueeze_(0)  # 32x25 -> 1x32x25 . 1 is for one hidden-layer (not-stacked)
        # logger.debug(f'h_sem1.h_sty0 {h_sem1.shape} {h_sty0.shape,merged.shape}')


        recon_sent0, _, _ = models.decoder(inputs=recon_target,  # pass not None for teacher focring  (batch, seq_len, input_size)
                                    encoder_hidden=merged,  # (num_layers * num_directions, batch_size, hidden_size)
                                    encoder_outputs=None,  # pass not None for attention
                                    teacher_forcing_ratio=1
                                    # in(0, 1-random.random()* epoch * 0.1) #range 0..1 , must pass inputs if >0. as epochs increase, it's lower
                                    )
        # print('$'*10,'recon_sent0 length',len(recon_sent0),'each',recon_sent0[0].shape)
        # rec_loss = nn.MSELoss()(recon_sent0,sent0)
        # see impl https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/loss/loss.py
        acc_loss, norm_term = 0, 0
        # logger.debug(f'recon_target {recon_target.shape} while decoder outputs {len(recon_sent0)}')
        # target shape [32, 26]  batch x words , actual len of 50
        for step, step_output in enumerate(recon_sent0):
            batch_size = recon_target.size(0)
            # print('step_output at step ',step,step_output.shape,type(step_output))
            outputs = step_output  # step_output.contiguous().view(batch_size, -1)

            # what to do if output is too-long? decision here is to match only relevant parts
            if step + 1 >= recon_target.size(1):
                # print ('breaking!!! at step',step)
                break
            gold = recon_target[:, step + 1]  # tuple [0] is data. [1] is len
            # print('output at step ',step,outputs.shape,type(outputs),'gold',gold.shape,type(gold))
            curr_loss = nll(outputs, gold)
            # logger.debug(f'loss for token {norm_term},{outputs.shape}, {gold.shape}, {curr_loss}')
            acc_loss += curr_loss
            norm_term += 1
        rec_loss = acc_loss / norm_term

        ######### ADV LOSS #########  TODO : willl it be better to use completely different sentences?
        h_sty1 = models.en_sty(b.sent_1)
        h_styX = models.en_sty(b.sent_x)
        adv_disc_p = models.adv_disc(torch.cat([h_sty1, h_styX], dim=merge_dim))
        # logger.debug (N(adv_disc_p[0:3].data).T)
        adv_target = T(torch.FloatTensor(np.full(shape=(b.sent_0[0].shape[0], 1), fill_value=0.5)))
        # the loss below is a parabula with min at log(0.5)=0.693... see documentation above
        adv_disc_loss = bce(adv_disc_p, adv_target) + np.log(0.5)  # np.log(0.5)=-0.693 ,
        # logger.debug(f'### adv_disc_loss {N(adv_disc_p.data[:5]).T} target={N(adv_target.data[:5]).T} bce={adv_disc_loss}')
        # logger.debug(f'    sanity test: on first step, you expect adv_disc_loss to be near zero')


        ######### BACKWARD #########
        # full loss
        loss = rec_loss + sim_loss * opt.sem_sim_weight + opt.sd_weight * adv_disc_loss  # rec_loss + sim_loss + opt.sd_weight*adv_disc_loss
        loss.backward()

        if not dont_optimize:
            optimizer_en_sem.step()
            optimizer_en_sty.step()
            optimizer_decoder.step()

        return N(sim_loss.data) * opt.sem_sim_weight, (rec_loss.data), N(adv_disc_loss.data) * opt.sd_weight  # N


    def train_scene_discriminator(b,models,dont_optimize=False):


        sent0 = T(b.sent_0)
        sent1 = T(b.sent_1)
        sentX = T(b.sent_x)
        y = T(b.is_x_0)

        models.adv_disc.zero_grad()

        merge_dim=1
        # h_sty0    = en_sty(sent0)
        h_sty0 = models.en_sty(sent1)
        h_sty0or2 = models.en_sty(sentX)  # same style, same or different semantics with random chance
        merged = torch.cat([h_sty0, h_sty0or2], dim=merge_dim)

        logger.debug(f'merged {merged.shape}')  # 4x32xdim
        out = models.adv_disc(merged)  #
        out = out.flatten()

        # TODO: #Note BCELossWithLogits is faster and more stable, to use it remove sigmoid from network end
        logger.debug(f'out {out.shape} y {y.shape}')

        # bce = nn.BCELoss()(out.flatten(), y.flatten()) #should wrapp in varaible?
        bce = bce2(out, y)  # should wrapp in varaible?
        # logger.debug('train_scene_discriminator {out} {y}')


        bce.backward()

        if not dont_optimize:
            optimizer_adv_disc.step()



        # print (out.shape) #torch.Size([16, 1])
        acc = np.round(N(out.detach())) == np.round(N(y))  # CHECK THIS DIMENSTIONS!!!
        logger.debug(f'adv_disc out {out.shape} is_x_0 {y.shape}')
        logger.debug(f'out {out.flatten()} y {y.flatten()} acc {acc} bce {bce.data}')
        # print (acc.shape) #1,16
        acc = acc.reshape(-1)  # .float()
        acc = acc.sum() / len(acc)
        return N(bce.data), N(acc)


    def one_epoc(epoch,bucket_iter_train,models,dont_optimize=False):
        logger.info('one_epoc starts')


        epoch_sim_loss, epoch_rec_loss, epoch_anti_disc_loss, epoch_sd_loss, epoch_sd_acc = 0, 0, 0, 0, 0

        training_batch_generator= None
        for i in range(opt.epoch_size if not dont_optimize else opt.epoch_size/10):
            # if i % 10==0 : print ('batch',i,'of',opt.epoch_size)
            logger.debug('next batch')
            b = None
            try:
                b = next(training_batch_generator)
            except:
                logger.info('creating new iterator')
                training_batch_generator = iter(bucket_iter_train)  # only if no choice... it's 1.5 min
                b = next(training_batch_generator)

            logger.debug('next batch is out')

            # train scene discriminator
            logger.debug(f'b_sent_0 {b.sent_0[0].shape}{b.sent_0[1].shape}')  # TEXT.reverse(b.sent_0))

            # %time train_scene_discriminator(b) #50ms
            sd_loss, sd_acc = train_scene_discriminator(b,models)

            logger.debug('train_scene_discriminator done')
            logger.debug('train_scene_discriminator done')
            epoch_sd_loss += sd_loss
            epoch_sd_acc += sd_acc

            # train main model
            # %time train(b,epoch) #300ms
            sim_loss, rec_loss, anti_disc_loss = train(b, models,epoch)
            logger.debug('train done')

            epoch_sim_loss += sim_loss
            epoch_rec_loss += rec_loss
            epoch_anti_disc_loss += anti_disc_loss

            logger.setLevel(logging.INFO)
        logger.info('[%02d] rec loss: %.4f | sim loss: %.4f | anti_disc_loss: %.4f || scene disc %.4f %.3f%% ' % (
        epoch, epoch_rec_loss / opt.epoch_size,
        epoch_sim_loss / opt.epoch_size, epoch_anti_disc_loss / opt.epoch_size,
        epoch_sd_loss / opt.epoch_size, 100 * epoch_sd_acc / opt.epoch_size))


    # --------- training loop ------------------------------------
    from eval import eval_sample
    from torch.optim.optimizer import Optimizer



    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
    logger.setLevel(logging.INFO)  # not DEBUG

    print ('running train with options:',opt)
    bucket_iter_train, _ = build_bible_datasets(verbose=False)
    models = build_models(bucket_iter_train.dataset, opt)

    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam



    models.en_sty.train()  # and not eval() mode
    models.en_sem.train()
    models.decoder.train()
    models.adv_disc.train()


    epoc_count = 0

    opt.lr = 0.001
    for lr in [opt.lr, opt.lr, opt.lr / 4, opt.lr / 4]:  # start 0.0001 for 2 big epocs
        print('lr', lr, opt.beta1)
        optimizer_en_sem = optimizer(models.en_sem.parameters(), lr, betas=(opt.beta1, 0.999))
        optimizer_en_sty = optimizer(models.en_sty.parameters(), lr, betas=(opt.beta1, 0.999))
        optimizer_decoder = optimizer(models.decoder.parameters(), lr, betas=(opt.beta1, 0.999))
        optimizer_adv_disc = torch.optim.SGD(models.adv_disc.parameters(), opt.adv_disc_lr)

        eval_sample(bucket_iter_train,models)
        for epoch in range(0, opt.epocs):
            one_epoc(epoc_count,bucket_iter_train,models)
            epoc_count += 1

    print('training loop done')

    # TODO: save the model
    # converge to 0.5 recon, but 1.6 adv loss: sd_weight !!! for lr in [opt.lr/4,opt.lr/4,opt.lr/8,opt.lr/8,opt.lr/16,opt.lr/16]: beta=0.5
    # problem: Hidden tied to embedding size. not logical (sem need more 1000. style need less 1)

#train_main(opt = get_options(False))