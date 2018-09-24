import argparse
import sys

def get_options(from_sysargv=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--epocs', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=500, help='epoch size')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')

    parser.add_argument('--optimizer', default='adam', help='optimizer to train with. only Adam supported.')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate. 0.001 is a food value')
    parser.add_argument('--adv_disc_lr', default=0.01, type=float, help='learning rate')

    parser.add_argument('--beta1', default=0.5, type=float, help='momentum term for adam')

    parser.add_argument('--semantics_dim', type=int, default=256, help='size of the semantics vector.default 256')
    parser.add_argument('--style_dim', type=int, default=1, help='size of the style vector.default 1') # was 60
    parser.add_argument('--sd_weight', type=float, default=1, help='weight on adversarial loss 0.0001 originally. 0.5 is good value!')
    parser.add_argument('--sem_sim_weight', type=float, default=1, help='weight on semantic similiarity loss')

    parser.add_argument('--max_sent_len', type=int, default=30, help='max size of sentence. sentences typically will be shorter')

    if not from_sysargv:
        sys.argv=["nothing"]
        print ('using default options')
    opt = parser.parse_args()
    return opt