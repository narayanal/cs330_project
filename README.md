CS330 Project

Steps to run the code

For generating GINC dataset, download the code ginc dataset code[1], and install all dependencies and run the following program
$python generate_data.py --n_hmms 1000 --n_train_samples 256 
no. of concepts ( is n_hmms / 2 ) is involved in generating the training code.
n_train_samples is the required no. of samples/documents generated.
Keep all other parameters as default.
GINC dataset is generated in specified output folder data/GINC_trans0.1_start10.0_nsymbols50_nvalues10_nslots10_vic0.9_nhmms10/train.json

For generating pretrain data,
Run the ginc dataset generation program with n_hmms = 20, 100, 200, 1000, 2000, 10000, 20000 and n_samples = 256

For generating validation data,
Run the ginc dataset generation program with n_hmms = 20000 and n_samples = 32





References:
[1] https://github.com/p-lambda/incontext-learning


