from train import train_main
from options import get_options
import sys
#start here
sys.argv = ["first_is_filename","--epoch_size","12","--dataset","bible"]
bucket_iter_train, bucket_iter_val, models= train_main(opt = get_options(True))

