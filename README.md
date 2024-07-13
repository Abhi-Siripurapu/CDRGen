# Antibody-Antigen Interaction Predictor

## Overview
This project leverages a fine-tuned protein language model to generate predicted sequences of antibodies with on average 120% better binding affinity (computationally determined, will experimentally verify soon!).

## Features
- **Sequence Generation**: Utilizes a fine-tuned model to generate antibody sequences.
- **Structure Prediction**: Uses AlphaFold to predict the 3D structure of generated sequences.
- **Affinity Prediction**: Employs the ANTIPASTI model to estimate the binding affinity from the DCCM of the predicted structure.
- **Screening Mechanism**: Screens multiple sequences to identify those with potentially higher affinities, enhancing therapeutic efficacy.


[ANTIPASTI](https://github.com/kevinmicha/ANTIPASTI)
CDRGen uses a custom version of the ANTIPASTI architecture that I trained myself, huge thanks to their team for making their data public, I essentially used the same things. ALso built on the Alphafold experiments they did in their research.


## Installation and Usage

Clone this repository to your local machine, then run: 
python cdr_gen.py <antibody_sequence> <antigen_sequence> <number_of_candidates>
Use one letter amino acids with spaces between each aa. Make sure you have the requirements in requirements.txt as well.




