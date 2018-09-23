import torch.nn as nn

from seq2seq.models import EncoderRNN, DecoderRNN
from torch import nn
from torch.nn import functional as F
from collections import namedtuple
import torch
from util import T,N

class BaseRNN(nn.Module):
    r"""
    Applies a multi-layer RNN to an input sequence.
    Note:
        Do not use this class directly, use one of the sub classes.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): maximum allowed length for the sequence to be processed
        hidden_size (int): number of features in the hidden state `h`
        input_dropout_p (float): dropout probability for the input sequence
        dropout_p (float): dropout probability for the output sequence
        n_layers (int): number of recurrent layers
        rnn_cell (str): type of RNN cell (Eg. 'LSTM' , 'GRU')
    Inputs: ``*args``, ``**kwargs``
        - ``*args``: variable length argument list.
        - ``**kwargs``: arbitrary keyword arguments.
    Attributes:
        SYM_MASK: masking symbol
        SYM_EOS: end-of-sequence symbol
    """
    SYM_MASK = "MASK"
    SYM_EOS = "EOS"

    def __init__(self, vocab_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.dropout_p = dropout_p

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class MyEncoderRNN(BaseRNN):
    """
    SMALL MODIFICATION ON SEQ2SEQ (hidden-state <> embedding)
    Applies a multi-layer RNN to an input sequence.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encodr (defulat False)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)
        embedding (torch.Tensor, optional): Pre-trained embedding.  The size of the tensor has to match
            the size of the embedding parameter: (vocab_size, hidden_size).  The embedding layer would be initialized
            with the tensor if provided (default: None).
        update_embedding (bool, optional): If the embedding should be updated during training (default: False).
    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`
    Examples::
         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)
    """

    def __init__(self, vocab_size, max_len, hidden_size, embed_size,
                 input_dropout_p=0, dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru', variable_lengths=False,
                 embedding=None, update_embedding=True):
        super(MyEncoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                                           input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, embed_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.rnn = self.rnn_cell(embed_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

    def forward(self, input_var, input_lengths=None):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden




########################## MODELS ##########################
class EncoderWrapper(nn.Module):
    """ wraps encoder, accpet in forward tuple of (data,len). return last hidden"""

    def __init__(self, encoder):
        super(EncoderWrapper, self).__init__()
        self.encoder = encoder

    def forward(self, inp):
        # in_data,in_len = in_tuple
        output, hidden = self.encoder(*inp)  # in_data,in_len)
        # **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        # **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`
        # return hidden[0,:,:]
        # return hidden[:,:,:].# view(1,hidden.size(1),-1)[0,:,:] #BUG BUG BUG O: check dim order, 2xbsxdim -> 1xbsxdim*2 ??

        # in lstm hidden is a tuple
        return torch.sum(hidden, dim=0)
        # TODO : BUG HERE


Models =  namedtuple("Models",["en_sem","en_sty","decoder","adv_disc"])

# Arch. Question How to work with bidi and multiple layers?
# currently, encoder combines with + (sum) bidi vectors. after 8 epocs of 100 batches(100 each) less then 1 recon_loss
# multiple layers not supported
def build_models(train_dataset, opt):

    #hardcoded, can't be changed in opt, as will break model
    variable_lengths = False  # True means batch is ordered. this can't be done as sent0.len!=sent1.len, to make it happen need to seperate batches!!!
    encoder_bidi = True
    decoder_bidi = True  # not supported True
    encoder_layers = 1  # not supported>1
    decoder_layers = 1

    TEXT = train_dataset.fields['sent_0']
    TEXT_TARGET = train_dataset.fields['sent_0_target']



    embed_size = TEXT.vocab.vectors.shape[1]
    en_sem = EncoderWrapper(
        MyEncoderRNN(len(TEXT.vocab), opt.max_sent_len, hidden_size=opt.semantics_dim, embed_size=embed_size,
                     variable_lengths=variable_lengths,
                     bidirectional=encoder_bidi, n_layers=encoder_layers,
                     input_dropout_p=0.1, dropout_p=0.0, rnn_cell='gru',
                     embedding=TEXT.vocab.vectors,  # (vocab_size, hidden_size)
                     update_embedding=True))

    en_sty = EncoderWrapper(
        MyEncoderRNN(len(TEXT.vocab), opt.max_sent_len, hidden_size=opt.style_dim, embed_size=embed_size,
                     variable_lengths=variable_lengths, bidirectional=encoder_bidi, n_layers=encoder_layers,
                     input_dropout_p=0.1, dropout_p=0.0, rnn_cell='gru',
                     embedding=TEXT.vocab.vectors,  # (vocab_size, hidden_size)
                     update_embedding=False))  # let only encoder do this

    decoder = DecoderRNN(len(TEXT.vocab), opt.max_sent_len,
                         (1 if encoder_bidi else 1) * (opt.semantics_dim + opt.style_dim), sos_id=TEXT_TARGET.sos_id,
                         eos_id=TEXT_TARGET.eos_id,
                         bidirectional=decoder_bidi, n_layers=decoder_layers,
                         input_dropout_p=0.1, dropout_p=0.0, rnn_cell='gru')
    # note here embedding are learnt (Which can be a waste too...)



    adv_disc = nn.Sequential(
        # input concat of two style
        nn.Linear(2 * opt.style_dim, 1),
        nn.Sigmoid()  # depends on what we have as loss #Must be sigmoid, we apply later BCE
    )

    strong_adv_disc = nn.Sequential(
        # input concat of two style
        nn.Linear(2 * opt.style_dim, 30),
        nn.PReLU(),
        nn.Linear(30, 20),
        nn.PReLU(),
        nn.Linear(20, 1),
        nn.Sigmoid()  # depends on what we have as loss #Must be sigmoid, we apply later BCE
    )

    en_sem = T(en_sem)
    en_sty = T(en_sty)
    decoder = T(decoder)
    adv_disc = T(adv_disc)

    return  Models(en_sem,en_sty,decoder,adv_disc)


def test():
    from torchtext import data
    from datasets import build_bible_datasets
    from options import get_options

    bucket_iter_train, _ = build_bible_datasets()
    sample = next(iter(bucket_iter_train))
    merge_dim = 1
    en_sem, en_sty, decoder, adv_disc = build_models(bucket_iter_train.dataset, get_options())
    print(type(sample.sent_0))
    in_var, in_len = sample.sent_0
    print(in_var.shape, in_len.shape)  # torch.Size([32, 56 or 66]) torch.Size([32])

    # print ('length0',sample.sent_0[1])
    # print ('length1',sample.sent_1[1])
    sem_out = T(en_sem(sample.sent_0))
    print('result of en_sem', sem_out.shape)  # [1, 32, 20]
    sty_out = T(en_sty(sample.sent_1))
    #print('sty_out', sty_out.shape, 'concat', T(torch.cat([sty_out, sty_out], dim=merge_dim)).shape)

    merged = T(torch.cat([sty_out, sty_out], dim=merge_dim))
    #print('merged1', merged.type(), merged.shape)
    disc_out = T(adv_disc(merged))
    #print(disc_out.shape)

    merged = T(torch.cat([sem_out, sty_out], dim=merge_dim))
    merged.unsqueeze_(0)
    #print('merged2', merged.shape)
    decoder_outputs, _, _ = decoder(inputs=None,  # pass not None for teacher focring  (batch, seq_len, input_size)
                                    encoder_hidden=merged,  # (num_layers * num_directions, batch_size, hidden_size)
                                    encoder_outputs=None,  # pass not None for attention
                                    teacher_forcing_ratio=0  # range 0..1 , must pass inputs if >0
                                    )
    decoder_outputs = decoder_outputs
    # **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
    #          the outputs of the decoding function.
    print('decoder_outputs', len(decoder_outputs), decoder_outputs[0].shape)
