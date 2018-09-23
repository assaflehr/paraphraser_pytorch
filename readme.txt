Installations:

Installation:  (requires few custom stages so no requirements.txt is provided)
!pip install --quiet torch -U  # 0.4 at-least, on windows, read installation instrcution of pytorch.org
!pip install --quiet torchtext  # for simpler datasets
!pip install  git+https://github.com/IBM/pytorch-seq2seq  #for seq2seq
!pip install --quiet dill  #req of seq2seq
!pip install --quiet tqdm  #req of seq2seq
!pip install --quiet revtok #for torchtext word tokenizer


datasets:
There are 3 avaialbes:
(1) toy: generated time strings in different formats
(2) quora questions pairs: download from kaggle site (requires registration)
    !pip install kaggle  #then create API key in the profile and fill it below
    !export KAGGLE_USERNAME="your-username"
    !export KAGGLE_KEY="your-key"
    !kaggle competitions download -c quora-question-pairs
    !kaggle competitions download -c quora-question-pairs
    !unzip train.csv.zip
    !unzip test.csv.zip
Some stats:
    !wc -l train.csv  #404302 train.csv
    !wc -l test.csv   #3563490 test.csv
example rows:
    "id","qid1","qid2","question1","question2","is_duplicate"
    "0","1","2","What is the step by step guide to invest in share market in india?","What is the step by step guide to invest in share market?","0"
    "1","3","4","What is the story of Kohinoor (Koh-i-Noor) Diamond?","What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?","0"
    "2","5","6","How can I increase the speed of my internet connection while using a VPN?","How can Internet speed be increased by hacking through DNS?","0"
    "3","7","8","Why am I mentally very lonely? How can I solve it?","Find the remainder when [math]23^{24}[/math] is divided by 24,23?","0"

(3) bible versions
    !wget https://github.com/scrollmapper/bible_databases/raw/master/csv/t_bbe.csv
    !wget https://github.com/scrollmapper/bible_databases/raw/master/csv/t_wbt.csv
    !head t_bbe.csv


Model structure:



# semantic_encoder : take sentence return a vector with only semantics
# semantic similiary objective: given a pair of sentences with the same semantics,
#    expect their results to be close (MeanSquared)

# see idea from https://github.com/edenton/drnet-py/blob/master/models/classifiers.py
# in her train method:
# FRAMES: VID A: FRAME 1 x_c1 , ?x_p2? (1/2 of times)
#                FRAME 2 x_c2 , x_p1
#         VID B: FRAME 1      , ?x_p2? (1/2 of times)
#
# x_p1,x_p2 chosen randonly from same/not-same video
# x_c1,x_c2 are always same-video
# c1 and p1 are same content, different pose.

# train_scene_discriminator()
# h_p1,h_p2 = netEP applied to x[0],x[1]. assumption: SAME VIDEO
# important: detach both!
# override half of h_p2 by random-permutations of the batch.
#   [1,2,3,4,5,6]
#   [2,3,1,4,5,6] after 1st half permute
#   [1,1,1,0,0,0] set unequal the labels (1=unequal, 0 equal. does it matter? should it be 0.9,0.1?)
# apply BCE on inpit: concat of [h_p1,h_p2]
# run backward, and optimizer on the netC classifier ONLY! emphasize! not on the encoder


# train()
# h_c1,h_c2 = netEC(x_c1), netEC(x_c2)  where input is x[0],x[1] sim loss is MSE directly on the hidden content-semantics
# h_p1,h_p2 = netEP(x_p1),netEP(x_p2) where input is x[2],x[3]
# rec = netD([h_c1, h_p1]) h_c1 is DIFFERENT FRAME , but same content, than h_p1
# netC is the semantic-discriminator given h_p1,h_p2, target 0.5 (max-entropy).
# emphasize! don't optimize netC is this stage

#Imp notes
# BIDI: when using bidi-encoder, we decided to merge the two D-dim vectors using +, getting D-dim embedding. empiracally, better result than concat
# OPTIMIZER: adv training is known to be unstable. We used the following known methods to make it more stable:
#            ??? sensativity to lr (fail to converge on close lr)- use AdamPre optimizer instead of Adam
#            ??? don't use ReLU layer, use Leaky instead (I used PreRELU)
#            never allow G or D to achieve high accuracy (before the end)
#            IMPORTANT: noisy input via dropout
#            IMPORTANT: label_smoothing on the ds
# ANTI-ADVERSERTIAL LOSS FUNCTION, target is 0.5, so min value is 0.6931471805599453
%matplotlib inline
import matplotlib.pyplot as plt
x = np.arange(1,100)/100.0
plt.title("anti adv-loss with target of 0.5. min is 0.693147 ")
plt.plot(x, -0.5*(np.log(x)+np.log(1-x)))
#-0.5*(np.log(0.53)+np.log(1-0.53))
