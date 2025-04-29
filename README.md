# TCS iON RIO125 - Automate Detection and Recognition of Grammatical Errors

This project focuses on detecting and correcting grammatical errors in English sentences using the T5 transformer model (from HuggingFace), enhanced with visualization and preprocessing techniques. It also includes a user-friendly Gradio interface for real-time correction.

---

## 🚀 Features

- Sentence and paragraph-level correction
- Error detection and highlighting
- Word cloud visualization of training data
- POS tagging and sentence length analysis
- Interactive web UI with Gradio
- Attention visualization using BertViz

---

## 📁 Dataset

The model uses the [JFLEG dataset](https://huggingface.co/datasets/jfleg), a benchmark dataset for fluency-oriented grammatical error correction. It includes:
- **Incorrect sentences** (`sentence`)
- **Multiple corrected versions** (`corrections`)

---

## 🧠 Model & Libraries

- Model: `t5-base` fine-tuned using HappyTransformer
- Tokenization: HuggingFace Transformers
- POS Tagging & Sentence Splitting: spaCy
- Visualization: WordCloud, Seaborn, Matplotlib, BertViz
- UI: Gradio

---

## 🛠️ Installation

```bash
!pip install spacy datasets bertviz wordcloud seaborn matplotlib
!pip install transformers==4.26.1 happytransformer==2.4.1
!pip install gradio
!python -m spacy download en_core_web_sm

```
---

## 📊 Preprocessing and Visualization

- Clean spacing and special characters
- Sentence-level parsing
- Word clouds of incorrect vs corrected tokens
- POS distribution analysis
- Sentence length frequency plots

---

## 📈 Training

Training is simulated using stored log values, but the real training pipeline uses:

```bash
from happytransformer import HappyTextToText, TTTrainArgs
g_model = HappyTextToText("T5", "t5-base")
arg = TTTrainArgs(batch_size=4, num_train_epochs=10)
g_model.train("train.csv", args=arg)

```

## Training Log Simulation

```bash
history = fit_train(g_model, 10, "train.csv", "val.csv", 4)
```
---

## Error Detection and Correction

The model can detect grammatical errors in a sentence and generate the corrected version. It can also highlight the differences between the original and corrected sentences using color-coded output.

---

## Evaluation

The model's performance is evaluated using accuracy and loss metrics. The training progress is visualized with plots showing the training and validation losses and accuracies over each epoch.

---

## Gradio Interface
A Gradio interface is created for easy interaction with the model. Users can input sentences, and the model will return the corrected version of the sentence in real-time.

---

## Usage

## Grammatical Error Detection

1. **Input a Sentence**: Enter a sentence that may contain grammatical errors.
2. **Correction**: The model will return a corrected version of the sentence.
3. **Visualization**: The differences between the original and corrected sentences are highlighted in color.

### Example Input:
- **Input Sentence**: "She go to the park regularly."
- **Corrected Sentence**: "She goes to the park regularly."

---

## Conclusion

This project demonstrates the power of T5 and transformer-based models for grammatical error correction. By leveraging datasets such as JFLEG and combining them with advanced techniques like spaCy and HappyTransformer, we aim to develop a robust and effective grammar correction system.

---

## Acknowledgments

- The **JFLEG dataset** is used in this project for training and testing the model.
- The **T5** model from Hugging Face is used for fine-tuning the grammatical error correction task.
- **spaCy** is used for tokenization, part-of-speech tagging, and other NLP tasks.

---
