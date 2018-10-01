1. Installation:  (It requires few custom stages so no requirements.txt is provided)
    !pip install --quiet torch -U  # 0.4 at-least, on windows, read installation instrcution of pytorch.org
    !pip install --quiet torchtext  # for simpler datasets
    !pip install  git+https://github.com/IBM/pytorch-seq2seq  #for seq2seq
    !pip install --quiet dill  #req of seq2seq
    !pip install --quiet tqdm  #req of seq2seq
    !pip install --quiet revtok #for torchtext word tokenizer

2. Download one of the datasets.

    (a) "quora" quora questions pairs: download from kaggle site (please register to the site, and thourgh your account, create api key)
        !pip install kaggle  #then create API key in the profile and fill it below
        import os
        os.environ['KAGGLE_USERNAME'] = "myusername"
        os.environ['KAGGLE_KEY'] = "mykey"
        !kaggle competitions download -c quora-question-pairs
        !unzip train.csv.zip

        example rows:  (404302 rows in total )
            "id","qid1","qid2","question1","question2","is_duplicate"
            "0","1","2","What is the step by step guide to invest in share market in india?","What is the step by step guide to invest in share market?","0"
            "1","3","4","What is the story of Kohinoor (Koh-i-Noor) Diamond?","What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?","0"

    (3b   !wget https://github.com/scrollmapper/bible_databases/raw/master/csv/t_bbe.csv
        !wget https://github.com/scrollmapper/bible_databases/raw/master/csv/t_wbt.csv
        !head t_bbe.csv


3. Usage:  (after installation and dataset download)
    Please see the samples at the "notebooks" folder for both command-line and API examples.

    example: python train.py --dataset quora

    see all the options, using "train.py --help"
    usage: train.py [-h] [--dataset DATASET] [--epocs EPOCS]
                    [--epoch_size EPOCH_SIZE] [--batch_size BATCH_SIZE]
                    [--optimizer OPTIMIZER] [--lr LR] [--adv_disc_lr ADV_DISC_LR]
                    [--beta1 BETA1] [--semantics_dim SEMANTICS_DIM]
                    [--style_dim STYLE_DIM] [--sd_weight SD_WEIGHT]
                    [--sem_sim_weight SEM_SIM_WEIGHT]
                    [--max_sent_len MAX_SENT_LEN]

    optional arguments:
      -h, --help            show this help message and exit
      --dataset DATASET     dataset to use. valid values are quora or bible
      --epocs EPOCS         number of epochs to train for
      --epoch_size EPOCH_SIZE
                            epoch size
      --batch_size BATCH_SIZE
                            batch size
      --optimizer OPTIMIZER
                            optimizer to train with. only Adam supported.
      --lr LR               learning rate. 0.001 is a good value
      --adv_disc_lr ADV_DISC_LR
                            learning rate
      --beta1 BETA1         momentum term for adam
      --semantics_dim SEMANTICS_DIM
                            size of the semantics vector.default 256
      --style_dim STYLE_DIM
                            size of the style vector.default 1
      --sd_weight SD_WEIGHT
                            weight on adversarial loss 0.0001 originally. 0.5 is
                            good value!
      --sem_sim_weight SEM_SIM_WEIGHT
                            weight on semantic similiarity loss
      --max_sent_len MAX_SENT_LEN
                            max size of sentence. sentences typically will be
                            shorter
