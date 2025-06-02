# Chinese Calligraphy Style Classification with ResNet152

This project applies a deep convolutional neural network (ResNet152) to classify handwritten Chinese characters into three major calligraphy styles: *Regular Script (楷書)*, *Running Script (行書)*, and *Clerical Script (隸書)*. The model is based on transfer learning using a pretrained ResNet152 architecture and fine-tuned for this multi-class image classification task.

## 🔍 Motivation

Although this project focuses on artistic handwriting recognition, the underlying techniques—such as image preprocessing, transfer learning, and hyperparameter tuning—are highly transferable to quantitative finance tasks, especially those involving pattern recognition, signal classification, or feature extraction in non-structured data.

## 🧠 Model

We use `tf.keras.applications.ResNet152` as the backbone. Key training details include:

- **Architecture**: ResNet152 (pretrained on ImageNet), with top layers customized for 3-class classification
- **Optimizer**: Adam
- **Loss function**: Categorical Crossentropy
- **Hyperparameters**:
  - Batch size: 32 / 64
  - Learning rate: 0.0001 / 0.0005
  - Epochs: 100
- **Input size**: 224 × 224 grayscale images (preprocessed from handwritten samples)

## 📈 Results

| Model      | Accuracy (Test) |
|------------|-----------------|
| Simple CNN | ~95.8%          |
| VGG16      | ~99.7%          |
| ResNet152  | **~99.9%**      |

ResNet152 showed the best performance in classifying ambiguous handwritten samples, especially where style features are subtle or overlapping.

## 🧩 Dataset

We used a combination of:
- Manually collected and preprocessed images of calligraphy
- Publicly available font libraries
- Handwritten character sets from classical Chinese calligraphers

You may adapt this pipeline with any Chinese character dataset of your choice for replication.

## 🧠 Reflection

Working on handwritten calligraphy classification helped me deeply understand the nuances of transfer learning and convolutional networks. These insights are particularly relevant for AI-driven financial modeling, such as identifying trading patterns or classifying time series regimes via transformed images or embeddings.

## 📌 Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy, Matplotlib, PIL
