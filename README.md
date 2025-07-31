# MNIST Artificial Neural Network from Scratch

A simple implementation of a neural network built from scratch using NumPy to classify handwritten digits from the MNIST dataset. This project demonstrates the fundamentals of neural networks including forward propagation, backpropagation, and gradient descent without using any deep learning frameworks.

## Features

- **Neural Network Architecture**: 2-layer neural network with ReLU activation and softmax output
- **Built from Scratch**: Implementation uses only NumPy, no TensorFlow/PyTorch
- **MNIST Classification**: Classifies handwritten digits (0-9)
- **Visualization**: Displays sample predictions with actual digit images
- **Performance Tracking**: Monitors training accuracy and validation performance

## Neural Network Architecture

- **Input Layer**: 784 neurons (28×28 pixel images flattened)
- **Hidden Layer**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons with softmax activation (for digits 0-9)

## Requirements

Before running the notebook, ensure you have the following Python packages installed:

```bash
pip install numpy pandas matplotlib
```

### Package Versions
- `numpy`: For numerical computations
- `pandas`: For data loading and manipulation
- `matplotlib`: For plotting and visualization

## Dataset

This project requires the MNIST dataset in CSV format. You can download it from:
- [Kaggle MNIST Dataset](https://www.kaggle.com/c/digit-recognizer/data)
- The dataset should contain pixel values (0-255) with the first column as labels

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nishagii/MNIST_data_classification_ANN.git
   cd MNIST_data_classification_ANN
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy pandas matplotlib
   ```

3. **Download the MNIST dataset**:
   - Download the MNIST dataset in CSV format
   - Place it in an accessible location on your system

4. **Update the dataset path**:
   - Open `MNIST_ANN_from_scratch.ipynb`
   - In the second cell, update the path to your MNIST dataset:
   ```python
   data = pd.read_csv('path/to/your/mnist/dataset.csv')
   ```

## How to Run

1. **Open the notebook**:
   ```bash
   jupyter notebook MNIST_ANN_from_scratch.ipynb
   ```
   Or use VS Code, Google Colab, or any Jupyter-compatible environment.

2. **Run cells sequentially**:
   - Start from the first cell and run each cell in order
   - The notebook is designed to be executed step-by-step

3. **Key execution steps**:
   - **Data Loading**: Load and preprocess the MNIST dataset
   - **Data Splitting**: Split into training (59,000) and validation (1,000) sets
   - **Model Training**: Train the neural network for 500 iterations
   - **Evaluation**: Test on validation set and visualize predictions

## Expected Results

- **Training Accuracy**: ~90% after 500 iterations
- **Validation Accuracy**: Similar performance on the dev set
- **Training Time**: A few minutes on a standard laptop
- **Visualization**: Sample digit images with predicted labels

## Code Structure

### Core Functions

- `init_params()`: Initialize weights and biases
- `forward_prop()`: Forward propagation through the network
- `backward_prop()`: Backpropagation to compute gradients
- `gradient_descent()`: Main training loop
- `make_predictions()`: Generate predictions for new data
- `test_prediction()`: Visualize individual predictions

### Key Components

1. **Activation Functions**:
   - ReLU for hidden layer
   - Softmax for output layer

2. **Loss Function**: Cross-entropy (implicit in softmax derivative)

3. **Optimization**: Gradient descent with learning rate 0.10

## Customization

You can experiment with different hyperparameters:

- **Learning Rate**: Modify `alpha` in `gradient_descent()` (default: 0.10)
- **Hidden Layer Size**: Change the first parameter in `init_params()` (default: 128)
- **Training Iterations**: Modify `iterations` parameter (default: 500)
- **Training/Validation Split**: Adjust the split ratio in data preprocessing

## Troubleshooting

### Common Issues

1. **Dataset Path Error**:
   - Ensure the CSV file path is correct
   - Check that the file exists and is readable

2. **Memory Issues**:
   - Reduce the training set size if you encounter memory problems
   - Consider using a smaller hidden layer

3. **Poor Performance**:
   - Try different learning rates (0.01, 0.1, 0.5)
   - Increase the number of iterations
   - Ensure data is properly normalized (divided by 255)

## Learning Objectives

This project demonstrates:
- Neural network fundamentals
- Matrix operations in neural networks
- Gradient descent optimization
- Forward and backward propagation
- Activation functions (ReLU, Softmax)
- One-hot encoding for classification
- Model evaluation and visualization

## Future Enhancements

Potential improvements to explore:
- Add more hidden layers (deep network)
- Implement different activation functions
- Add regularization (L1/L2, dropout)
- Use different optimizers (Adam, RMSprop)
- Add batch processing for larger datasets
- Implement early stopping

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to fork this repository and submit pull requests for any improvements or bug fixes.

## Author

Created by [nishagii](https://github.com/nishagii)

---

**Note**: This is an educational project designed to understand neural network fundamentals. For production use, consider using established deep learning frameworks like TensorFlow or PyTorch.

