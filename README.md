CS330 Project

**Steps to generated GINC dataset**

1. For generating GINC dataset, download the ginc dataset code[1], and install all dependencies and run the following program
   a. $python generate_data.py --n_hmms 1000 --n_train_samples 256 
   b. no. of concepts ( is n_hmms / 2 ) is involved in generating the training code.
   c. n_train_samples is the required no. of samples/documents generated.
   d. Keep all other parameters as default.
   e. GINC dataset is generated in specified output folder data/GINC_trans0.1_start10.0_nsymbols50_nvalues10_nslots10_vic0.9_nhmms10/train.json

2. For generating pretrain data,
Run the ginc dataset generation program with n_hmms = 20, 100, 200, 1000, 2000, 10000, 20000 and n_samples = 256

3. For generating validation data,
Run the ginc dataset generation program with n_hmms = 20000 and n_samples = 32

**Steps to run the code**
1. For running pre-training and validation of GINC dataset, download the code [2], and install all dependencies.
2. Add the local folder path of /src/diversity in ginc_train_eval.py
     import sys
     sys.path.append('/home/jupyter/project/beyond-scale-language-data-diversity/src/diversity')
     sys.path.append('/home/jupyter/project/beyond-scale-language-data-diversity/src')
3. Specify the variables ginc_dataset_root folder and val_file, in ginc_train_eval.py
     ginc_dataset_root = "../ginc-output-repro/data/"
     val_file = '../ginc-output-repro/val/GINC_trans0.1_start10.0_nsymbols13_nvalues12_nslots11_nsamples32_nhmms20000_seed1111/train.json'
4. Run the code $python ginc_train_eval.py
5. 



References:
[1] https://github.com/p-lambda/incontext-learning
[2] https://github.com/alycialee/beyond-scale-language-data-diversity

