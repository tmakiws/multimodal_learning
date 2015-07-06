# Multimodal Learning

## Usage
   For pascal sentence dataset, simply :`python multi_pascal.py`

### optional arguments
  -h, --help:       show this help message and exit
  
  -m, --model:      1: ML-ELM, 2: HeMap

  -j, --joint:      joint method (cca, pcca)
  
  --hidden:         number of hidden units
  
  --output:         number of output units
  
  -r, --reg:        regularize parameter of feature extraction layers
  
  -c, --cca:        regularize parameter of cca
  
  -l, --layers:     number of layers
  
  -f, --file:       filename
  
-q, --quiet:        do not print to file
