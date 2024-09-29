# HP-RDTS

This repository introduces a high-performance Raman distributed temperature sensing (RDTS) system that leverages a 1D deep convolutional neural network (1DDCNN) to significantly reduce temperature uncertainty.

## Key Features

- **1D-CNN**:  We utilize a 1D Convolutional Neural Network (CNN) architecture to enhance the signal-to-noise ratio (SNR) and reduce temperature uncertainty in the temperature sensing algorithm. This allows for more accurate temperature measurements along the fiber.

- **Improved Signal-to-Noise Ratio (SNR):** The 1DDCNN effectively removes noise from the SpRS signal, improving the SNR and leading to more accurate temperature measurements.

- **Improved Accuracy in Long-Distance Sensing:** The 1DDCNN maintains its denoising effectiveness even with signal attenuation, enhancing accuracy in long-distance temperature sensing.

- **Reduced Complexity and Cost:** The 1DDCNN implementation is relatively simple, eliminating the need for specialized optical fibers or pulse coding techniques, which reduces system complexity and cost.

## Repository Contents

This repository provides the necessary codes for the evaluation and simulation of the proposed scheme. It includes:

- **Training and Testing Codes**: You will find model architecture, and both training and testing codes in net directory.

- **RDTS Model**: We have included the code for creating RDTS model for evaluation of our method in data directory.

- **Trained Models**: Pre-trained models and simulation data are provided for easy evaluation and testing purposes, which are in the result directory.

We encourage you to explore this repository and contribute to this aspect.

Thank you for your interest in the HP-RDTS repository. 

If you have any questions or need further assistance, don't hesitate to reach out.
Contact: Hao Wu (wuhaoboom@hust.edu.cn)

Best regards. 