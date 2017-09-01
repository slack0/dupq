# Detecting Quora Duplicates

### Getting started

- Code tested and working in Anaconda distribution
- Recommended to create a new conda environment with python==2.7
- With the included requirements.txt do a 'pip install requirements.txt'
- This code uses GloVe embeddings from Stanford NLP. You can either pre-download, unzip and move it to $PWD/data directory, or call download_data() routine. Refer to code for proper usage
- The file 'main.py' provides sample usage of the code. Refer to the main class in quora_dupl.py for details, if needed

### Some Notes/Limitations

- The GloVe embeddings currently used does not span the entire vocabulary of Quora dataset

- Training/evaluation was done on AWS P2 instance with a single K80 GPU. Training time noticed with current hyperparameter settings is ~150sec

- With ~20 epochs, accuracy on the test portion of data was observed to be ~0.80 - 0.81


### Recommendations

- It is *strongly* recommended to run this code using Tensorflow GPU. CPU training was observed to be very slow. Ensure proper GPU utilization by turning tf.Config() settings if necessary.
