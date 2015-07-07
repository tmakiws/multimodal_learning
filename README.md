# Multimodal Learning

## Usage
   For pascal sentence dataset, simply :`python multi_pascal.py`

### optional arguments
  -h, --help:       show this help message and exit
  
  -m, --model:      1: ML-ELM, 2: HeMap (default: 1)

  -j, --joint:      joint method (cca, pcca)
  
  --hidden:         number of hidden units (default: 256)
  
  --output:         number of output units (default: 64)
  
  -r, --reg:        regularize parameter of feature extraction layers (default: 1)
  
  -c, --cca:        regularize parameter of cca (default: 1)
  
  -l, --layers:     the number of iteration of (ELM/HeMap + CCA) (default: 1)
  
  -f, --file:       filename (default: result.txt)
  
  -q, --quiet:      do not print to file

### example
  pascal dataset retrieval used the ML-ELM model in which the number of hidden units is 512 and the number of layers is 5 (2 iterations of ELM+CCA)
  `python multi_pascal.py --hidden 512 -l 2`
