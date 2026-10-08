# AI Desire Detector Using Deep Learning

## 1. Abstract
The AI Desire Detector is a student demonstration of text classification using a neural network. It takes a sentence describing a user's goal, interest, or preference and predicts one of four categories: Learning & Career, Travel & Adventure, Food & Cooking, or Fitness & Health. The project demonstrates text preprocessing, tokenization, sequence padding, neural-network training, and a simple web interface.

## 2. Problem Statement
People express interests and goals through natural language. The aim of this project is to build a basic application that categorizes the topic expressed in a short sentence.

## 3. Objectives
- Learn the basics of deep learning for text classification.
- Convert text into numeric sequences.
- Train a neural network to predict a category.
- Display the prediction through a simple web interface.
- Understand the limitations of small datasets and model confidence.

## 4. Tools and Technologies
- Python
- TensorFlow / Keras
- NumPy
- scikit-learn
- Streamlit
- CSV dataset

## 5. Methodology
1. Load labelled example sentences from a CSV file.
2. Convert category labels to numeric values.
3. Tokenize words and convert sentences into sequences.
4. Pad sequences to a fixed length.
5. Train an embedding-based neural network with a softmax output layer.
6. Evaluate the model on a held-out validation split.
7. Use Streamlit to accept a sentence and display predicted category scores.

## 6. System Architecture
User sentence → Tokenizer → Sequence padding → Trained neural network → Category probabilities → Streamlit result screen.

## 7. Model Description
The model contains an Embedding layer, Global Average Pooling, a fully connected ReLU layer, Dropout, and a Softmax output layer. The output layer has one unit per category. The model is trained using Adam and sparse categorical cross-entropy.

## 8. Dataset
The included CSV contains short, labelled example sentences. It is a toy dataset for classroom demonstration and is not representative of all users or language styles.

## 9. Expected Output
For a sentence such as “I want to learn Python and build useful apps,” the model should generally predict Learning & Career. Predictions can be wrong, especially for sentences unlike the training examples.

## 10. Limitations
- Small dataset and limited vocabulary.
- Only four predefined categories.
- Performance depends on training examples and wording.
- A model score is not proof of a person's true or hidden desire.
- The model may perform poorly on slang, spelling errors, or mixed-category sentences.

## 11. Future Scope
- Expand the dataset with consent and diverse writing styles.
- Add more categories and multilingual support.
- Report precision, recall, F1-score, and a confusion matrix.
- Compare a neural network with a TF-IDF baseline.
- Add a feedback feature to correct predictions.

## 12. Conclusion
This project demonstrates a complete beginner-level deep-learning workflow for classifying expressed interests from text, from data preprocessing and training to a usable web interface. It should be treated as an educational prototype rather than a psychological assessment tool.
