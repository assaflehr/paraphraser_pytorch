from train import train_main
from options import get_options
import sys
#start here
sys.argv = ["first_is_filename","--epoch_size","3"]
train_main(opt = get_options(False))