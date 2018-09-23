import torch
import numpy as np
from util import revers_vocab,T,N


def eval_sample(bucket_iter,models,samples=3,):
    # back to eval mode
    models.en_sem.eval()
    models.en_sty.eval()  # and not eval() mode
    models.decoder.eval()
    models.adv_disc.eval()

    merge_dim=1


    TEXT = bucket_iter.dataset.fields['sent_0']
    TEXT_TARGET = bucket_iter.dataset.fields['sent_0_target']


    b = next(iter(bucket_iter))

    sent0 = T(b.sent_0)
    sent1 = T(b.sent_1)
    sentX = T(b.sent_x)

    recon_target = b.sent_0_target


    h_sem0 = models.en_sem(sent0)
    h_sem1 = models.en_sem(sent1)
    h_semX = models.en_sem(sentX)

    h_sty0 = models.en_sty(sent0)
    h_sty1 = models.en_sty(sent1)
    h_styX = models.en_sty(sentX)

    recon_sem0_sty0, _ ,_ = models.decoder(inputs=None,  # pass not None for teacher focring  (batch, seq_len, input_size)
                                    # encoder_hidden=T(torch.cat([h_sem1,h_sty0],dim=1)).unsqueeze(0), #(num_layers * num_directions, batch_size, hidden_size)
                                   encoder_hidden=T(torch.cat([h_sem0 ,h_sty0] ,dim=merge_dim)).unsqueeze(0)
                                    ,  # (num_layers * num_directions, batch_size, hidden_size)
                                   encoder_outputs = None,  # pass not None for attention
                                   teacher_forcing_ratio=0  # range 0..1 , must pass inputs if >0
                                    )


    recon_semX_sty0, _ ,_ = models.decoder(inputs=None,  # pass not None for teacher focring  (batch, seq_len, input_size)
                                    # encoder_hidden=T(torch.cat([h_sem1,h_sty0],dim=1)).unsqueeze(0), #(num_layers * num_directions, batch_size, hidden_size)
                                   encoder_hidden=T(torch.cat([h_semX ,h_sty0] ,dim=merge_dim)).unsqueeze(0)
                                    ,  # (num_layers * num_directions, batch_size, hidden_size)
                                   encoder_outputs = None,  # pass not None for attention
                                   teacher_forcing_ratio=0  # range 0..1 , must pass inputs if >0
                                    )

    recon_semX_sty1, _ ,_ = models.decoder(inputs=None,  # pass not None for teacher focring  (batch, seq_len, input_size)
                                   encoder_hidden=T(torch.cat([h_semX ,h_sty1] ,dim=merge_dim)).unsqueeze(0),
                                    # (num_layers * num_directions, batch_size, hidden_size)
                                   encoder_outputs = None,  # pass not None for attention
                                   teacher_forcing_ratio=0  # range 0..1 , must pass inputs if >0
                                    )
    merge_dim=1
    recon_semX_sty1_tf1, _ ,_ = models.decoder(inputs=recon_target,  # pass not None for teacher focring  (batch, seq_len, input_size)
                                       encoder_hidden=T(torch.cat([h_semX ,h_sty1] ,dim=merge_dim)).unsqueeze(0),
                                        # (num_layers * num_directions, batch_size, hidden_size)
                                       encoder_outputs = None,  # pass not None for attention
                                       teacher_forcing_ratio=1  # range 0..1 , must pass inputs if >0
                                        )
    seperator=' '
    for i in range(samples):
        print ('\n%20s'% 'sent0:', revers_vocab(TEXT_TARGET.vocab, sent0[0][i], seperator))
        print ('%20s'% 'sent0_targ:', revers_vocab(TEXT_TARGET.vocab, recon_target[i], seperator))
        print ('%20s'% 'sent1:', revers_vocab(TEXT_TARGET.vocab, sent1[0][i], seperator))
        print ('%20s'% 'sentX:', revers_vocab(TEXT_TARGET.vocab, sentX[0][i], seperator))

        # recon_sent is a list of : batch x softmax array
        tokens= ( np.array(([N(torch.argmax(l, dim=1)[i]) for l in recon_sem0_sty0])))
        print ('%20s'% 'recon_sem0_sty0:[TF=0]', revers_vocab(TEXT_TARGET.vocab, tokens, seperator))

        tokens= ( np.array(([N(torch.argmax(l, dim=1)[i]) for l in recon_semX_sty0])))
        print ('%20s'% 'recon_semX_sty0:[TF=0]', revers_vocab(TEXT_TARGET.vocab, tokens, seperator))

        tokens= ( np.array(([N(torch.argmax(l, dim=1)[i]) for l in recon_semX_sty1])))
        print ('%20s'% 'recon_semX_sty1:[TF=0]', revers_vocab(TEXT_TARGET.vocab, tokens, seperator))

        tokens= ( np.array(([N(torch.argmax(l, dim=1)[i]) for l in recon_semX_sty1_tf1])))
        print ('%20s'% 'recon_semX_sty1:[TF=1]', revers_vocab(TEXT_TARGET.vocab, tokens, seperator))

    models.en_sty.train()  # and not eval() mode
    models.en_sem.train()
    models.decoder.train()
    models.adv_disc.train()


