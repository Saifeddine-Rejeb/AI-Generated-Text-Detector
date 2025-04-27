# AI Generated Text Detector

## Description

This project is a text classifier designed to distinguish between AI-generated and human-generated text. It leverages the Hello-SimpleAI/HC3 dataset from Hugging Face and employs a Support Vector Machine (SVM) model for classification.

## Features

- **Dataset**: Utilizes the Hello-SimpleAI/HC3 dataset.
- **Model**: SVM classifier with two classes: AI-generated and human-generated text.
- **TF-IDF Vectorization**: Converts text data into numerical features for model training.
- **Model Persistence**: Saves the trained SVM model and TF-IDF vectorizer for future use.
- **Hyperparameter Tuning**: Includes grid search for optimizing SVM parameters.

## Features

- **Dataset**: Utilizes the Hello-SimpleAI/HC3 dataset.
- **Model**: SVM classifier with two classes: AI-generated and human-generated text.
- **TF-IDF Vectorization**: Converts text data into numerical features for model training.
- **Model Persistence**: Saves the trained SVM model and TF-IDF vectorizer for future use.
- **Hyperparameter Tuning**: Includes grid search for optimizing SVM parameters.

## Installation

1. Clone the repository.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Load the HC3 dataset using the provided code.
2. Preprocess the data by merging human and AI-generated answers.
3. Train the SVM model using the TF-IDF features.
4. Save the trained model and vectorizer for inference.
5. Use the saved model to classify new text samples.

## Example

### Input

```python
# Example text input
test_input = ["Exercise is beneficial for both physical and mental health."]
```

### Output

```python
# Predicted class
['human']
```

## Why Choose SVM?

Support Vector Machines (SVM) are particularly effective in high-dimensional spaces, making them a great choice for text classification tasks where the feature space (e.g., TF-IDF vectors) can be very large. SVMs are also robust to overfitting, especially in cases where the number of features exceeds the number of samples. The linear kernel was chosen for its simplicity and effectiveness in separating the two classes (AI-generated vs. human-generated text).

Additionally, SVMs provide flexibility in choosing different kernels, allowing us to experiment with non-linear decision boundaries if needed. The hyperparameter tuning using GridSearchCV further ensures that the model is optimized for the given dataset.

## Predictions

The trained SVM model was tested on unseen data, and the predictions were evaluated using precision, recall, and F1-score. The results indicate that the model performs well in distinguishing between AI-generated and human-generated text (96% on all metrics). Below are some example predictions:

- **Input**: "Honestly, exercise has so many upsides. It keeps your heart healthy, helps you stay in shape, and can even boost your mood when you're having a rough day."

  - **Predicted Class**: Human

- **Input**: "Exercise offers numerous benefits for both physical and mental health. Regular physical activity can help improve cardiovascular health, strengthen muscles, and enhance flexibility."
  - **Predicted Class**: AI-generated


## References

- [HC3 Dataset](https://huggingface.co/datasets/Hello-SimpleAI/HC3)
- [SVM Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

## Citation

Robust Detection of LLM-Generated Text: A Comparative Analysis  
[https://arxiv.org/html/2411.06248v1](https://arxiv.org/html/2411.06248v1)
