# TensorFlow Recurrent Network

A standalone Python export of a TensorFlow 2 LSTM that classifies MNIST digits by treating each image row as a sequence.

## Setup

The example uses the TensorFlow 2-era Keras API. Use Python 3.10 with the dependencies pinned in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
python recurrent_network.py
```

The script downloads MNIST through Keras on first run and trains for 1,000 steps, printing loss and accuracy every 100 steps. Internet access is required for the initial dataset download. Training starts when the script is run.

## Attribution

The source notebook credits Aymeric Damien and the [TensorFlow-Examples project](https://github.com/aymericdamien/TensorFlow-Examples/). This repository preserves that attribution while exporting the notebook's Python cells to a script.