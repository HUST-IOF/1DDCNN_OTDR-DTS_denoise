# HP-RDTS

Welcome to the HP-RDTS repository, which presents a high-performance Raman Distributed Temperature Sensing (RDTS) system utilizing a 1D Deep Convolutional Neural Network (1DDCNN) to significantly reduce temperature uncertainty.

## Key Features

- **1D-CNN**: Our architecture leverages a 1D Convolutional Neural Network to enhance the signal-to-noise ratio (SNR), allowing for more accurate temperature measurements along the fiber.

- **Enhanced Signal-to-Noise Ratio**: The 1DDCNN effectively filters noise from the SpRS signal, resulting in improved SNR and precision.

- **Long-Distance Accuracy**: The 1DDCNN preserves its denoising capabilities even with signal attenuation, enhancing accuracy in long-distance temperature sensing.

- **Reduced Complexity and Cost**: This implementation simplifies the system by eliminating the need for specialized optical fibers or complex pulse coding techniques, thereby lowering overall costs.

## Repository Contents

This repository includes essential codes for evaluating and simulating our proposed method, featuring:

- **Training and Testing Codes**: Find the model architecture and training/testing scripts in the `net` directory.

- **RDTS Model**: Code for creating the RDTS model is available in the `data` directory for method evaluation.

- **Trained Models**: Pre-trained models and simulation data for easy evaluation and testing are located in the `result` directory.

- **Research Paper**: The published paper is provided in the `paper` directory for your reference.

We invite you to explore this repository and contribute to advancing this research area.

For any questions or further assistance, please feel free to reach out.

Contact: Dr. Hao Wu (wuhaoboom@hust.edu.cn)

Best regards.