## About CGraphDTA

CGraphDTA is a fusion-based deep learning architecture for detecting drug-target binding affinity using target sequence and structure.  

The benchmark dataset can be found in `./data/`. The CGraphDTA model is available in `./src/`. And the result will be generated in `./results/`. See our paper for more details.

**[IMPORTANT] We provide the input file in the [release page](https://github.com/CSUBioGroup/CGraphDTA/releases/tag/Input).** Please download it to `./data/`.


### Software and database requirement  
To run the full version of CGraphDTA, you need to install the following three software and download the corresponding databases:  
[BLAST+](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) and [UniRef90](https://www.uniprot.org/downloads)  
[HH-suite](https://github.com/soedinglab/hh-suite) and [Uniclust30](https://uniclust.mmseqs.com/)  
[DSSP](https://github.com/cmbi/dssp)
[Mol2vec](https://github.com/samoturk/mol2vec)
Besides, RDKit 2019.09.3 is also need to change the format of drugs.


### Requirements:
- python 3.7.11
- pytorch 1.9.0
- scikit-learn 0.24.2
- dgl 0.9.1.post1
- tqdm 4.62.2
- ipython 7.27.0
- numpy 1.20.3
- pandas 1.3.2
- numba 0.53.1
- scipy 1.7.1
- einops 0.6.0
- loguru 0.6.0

### Installation

In order to get CGraphDTA, you need to clone this repo:

```bash
git clone https://github.com/CSUBioGroup/CGraphDTA
cd CGraphDTA
```
The easiest way to install the required packages is to create environment with GPU-enabled version:
```bash
conda env create -f environment_gpu.yml
conda activate CGraphDTA
```
### Predict

to use our model
```bash
cd ./src/
python predict.py
```

### Training

to train your own model
```bash
cd ./src/
python train.py
```

### contact
Kaili Wang: kailiwang@csu.edu.cn 
You can also download the codes from https://github.com/KailiWang1/CGraphDTA
