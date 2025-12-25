# lstm-hate-speech-detection

## 📌 Project Overview

This project addresses the problem of **hate speech detection on social media** using **deep learning**. The task is formulated as a **multi-class text classification problem**, where each tweet is classified into one of three categories:

* **Hate Speech (Class 0):** Explicit hateful or harmful content targeting individuals or groups.
* **Offensive Language (Class 1):** Abusive or offensive language without explicit hate intent.
* **Neither (Class 2):** Neutral content with no offensive or hateful language.

---

## 📊 Dataset

* **Source:** Hate Speech Dataset (CSV)
* **Total Samples:** 24,783 tweets
* **Key Challenge:** Severe **class imbalance**, where the *Offensive* class dominates the dataset.

### Class Distribution

* Offensive Language ≫ Neutral ≫ Hate Speech

This imbalance significantly affects model performance, especially recall and F1-score for the minority class (Hate Speech).

---

## 🧹 Text Preprocessing

The following preprocessing steps are applied:

* Lowercasing text
* Removing URLs, mentions, hashtags, emails and numbers
* Handling emojis
* Stopword removal
* Lemmatization using NLTK
* Tokenization and padding

---

## ⚖️ Handling Class Imbalance

Class imbalance is one of the main challenges of this dataset. Several strategies were implemented and compared directly in the training notebook:

1. **No Augmentation**

   * Original dataset used as-is
   * Model biased toward the majority (Offensive) class

2. **Upsampling**

   * Minority classes are augmented using text-level data augmentation techniques
   * Operations include synonym replacement, random deletion, and word swapping
   * Increasing data diversity rather than simple duplication helps improve generalization for minority classes while reducing overfitting



3. **Downsampling + Upsampling**

   * Majority class reduced while minority classes are increased
   * Produces a more balanced class distribution

4. **Class Weights (W)**

   * Higher loss penalty assigned to minority classes
   * Forces the model to pay more attention to underrepresented labels

5. **Combined Strategies (Aug + W)**

   * Balancing the dataset while also applying class weights

Each configuration was trained and evaluated using the same pipeline to allow fair comparison using **Accuracy, Precision, Recall, and F1-score**.

---

## 🧠 Model Architecture

The model used in this project is a **deep learning LSTM-based classifier** implemented using **TensorFlow / Keras**. The architecture is designed to capture sequential and contextual information in tweets.

**Architecture Details:**

* **Embedding Layer**
  * Trainable word embeddings learned during training
  * Converts tokenized text into dense vector representations
    
* **LSTM Layer**
  * Captures long-range dependencies and word order information
    
* **Dropout Layer**
  * Reduces overfitting by randomly disabling neurons during training
    
* **Dense Output Layer**
  * Fully connected layer with **Softmax activation**
  * Outputs probabilities for the 3 classes (Hate, Offensive, Neutral)

**Loss Function:** Categorical Cross-Entropy
**Optimizer:** Adam
**Evaluation Metrics:** Accuracy, Precision, Recall, F1-score

---

## 📈 Experimental Results (Summary)

Multiple experiments were conducted to evaluate the impact of data balancing and class weighting.

| Configuration                             | Accuracy | F1 (Hate)       | General Observation                      |
| ----------------------------------------- | -------- | --------------- | ---------------------------------------- |
| No Augmentation                           | High     | Low             | Model biased toward majority class       |
| No Aug + Class Weights                    | Lower    | Unstable        | Accuracy drops, minority recall improves |
| Upsampling                                | Moderate | Mixed           | Risk of overfitting                      |
| Upsampling + Class Weights                | Lower    | Slightly better | Still unstable                           |
| Up- & Down-Sampling                       | Best     | High            | Most balanced performance                |
| Up- & Down-Sampling  + Class Weights      | Stable   | High            | Best trade-off overall                   |

**Final Conclusion:**

> Accuracy alone is misleading for imbalanced datasets. The best-performing setup is **Downsampling + Upsampling with optional class weights**, as it achieves the highest and most stable F1-score for the Hate Speech class.

---

## 🧪 Evaluation

Model evaluation was performed using multiple metrics to properly reflect performance on imbalanced data:

* **Confusion Matrix** to visualize class-wise predictions
* **Classification Report** including precision, recall, and F1-score
  
Accuracy was not used as the sole metric because it is dominated by the majority class.

---
