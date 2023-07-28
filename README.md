# ProTact

Source code for "Accurate and Scalable Protein Interface Prediction through SE(3)-Invariant Geometric Graph Neural Networks"

## Install requirements

### PSAIA

Please follow the [DeepInteract#installing-psaia](https://github.com/BioinfoMachineLearning/DeepInteract#installing-psaia) to install the PSAIA tools while the ```psaia_config_file_input.txt``` is in ```ProTact/utils/datasets/builder```, and set the psaia_dir in ```ProTact/utils/data.py```.

### Genetic databases

In this work, the genetic databases are used to generate the protein complexes.
Follow the [DeepInteract#Genetic databases](https://github.com/BioinfoMachineLearning/DeepInteract#genetic-databases),
you can get the genetic databases easily.
But the genetic databases are not necessary for the prediction, you can use ```--no_fast``` to generate the protein complexes the genetic databases.
If you set the ```--no_fast```, you should change the ```ProTact/utils/data.py``` to set the ```hhsuite_db``` to ```your database path```.

### Environment

Python = 3.9

Python packages listed in environment.yml

To install all the Python packages, create a new conda environment:

```
conda env create -f environment.yml
conda activate ProTact
```
PS: If you want to use the GPU, you should install the CUDA version the same with your GPU driver support to make the PyKeops package work, in my opinion.

## Data

Our model can input the ```*.pdb``` files directly. The ```*.pdb``` files can be downloaded from [Protein Data Bank](https://www.rcsb.org/).

## Prediction

The ```predict_oneshot.py``` can be used to predict the interface residues of a protein complex. The ```predict_oneshot.py``` will output the predicted contact map in ```contact_map.pt``` format.

```
python predict_oneshot.py --left_pdb examples/1SDU_A.pdb --right_pdb examples/1SDU_B.pdb --model model/best_dips.pt --device cuda:0
```

## Training

We will release the training code and details soon.

