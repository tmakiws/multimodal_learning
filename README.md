# Multimodal Learning

## Usage
   For pascal sentence dataset, simply :`python multi_pascal.py`

### optional arguments
  -h, --help:            show this help message and exit
  
  -m, --model:           1: ML-ELM, 2: HeMap

  -j JOINT_METHOD, --joint JOINT_METHOD:
                        joint method
  
  --hidden HIDDEN:       number of hidden units
  
  --output OUTPUT:       number of output units

  -r REG_PARAM, --reg REG_PARAM:
                        regularize parameter of feature extraction layers
  
  -c CCA_PARAM, --cca CCA_PARAM:
                        regularize parameter of cca
  
  -l LAYERS, --layers LAYERS:
                        number of layers
  
  -f FILE, --file FILE:  filename
  
  -q, --quiet:           do not print to file
