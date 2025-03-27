# Azerbaijani Spell Checker

A deep learning-based spell-checking system for Azerbaijani language that uses Bidirectional LSTM neural networks to classify whether a word is spelled correctly or not.

## Overview

This project implements a neural spell checker that analyzes Azerbaijani words and classifies them as either correctly spelled (1) or misspelled (0). The system uses a ByteLevelBPE tokenizer and a Bidirectional LSTM architecture for accurate word classification.

## Model Architecture

The spell checker uses a neural network with the following components:

- **Tokenization**: ByteLevelBPE (Byte-Level Byte Pair Encoding) tokenizer for efficient character-level tokenization
- **Embedding Layer**: Converts tokenized inputs into dense vector representations
- **Bidirectional LSTM Layers**: Two stacked BiLSTM layers for capturing context in both directions
- **Dropout Layer**: Prevents overfitting by randomly dropping neurons during training
- **Output Layer**: Single neuron with sigmoid activation for binary classification

The BiLSTM architecture is particularly well-suited for this task as it can capture contextual patterns in both directions of the word, learning character dependencies that are important for spelling analysis.

## Performance

The model achieves strong performance metrics:

- **Accuracy**: 94.76%
- **Precision**: 
  - Correct words (class 0): 98%
  - Incorrect words (class 1): 82%
- **Recall**:
  - Correct words (class 0): 95%
  - Incorrect words (class 1): 92%
- **F1-Score**:
  - Correct words (class 0): 97%
  - Incorrect words (class 1): 87%

## Dataset

The model is trained on a dataset containing pairs of Azerbaijani words and their corresponding labels:

```
word,label
köybək,0
koynek,0
koynək,0
köynek,0
jöynək,0
löynək,0
ab-hava,1
ab-havalı,1
abadan,1
abadanlıq,1
abadi,1
abadlaşan,1
abadlaşdırma,1
abadlaşdırmaq,1
abadlaşdırılan,1
abadlaşdırılma,1
abadlaşdırılmaq,1
```

Where:
- `1` indicates a correctly spelled word
- `0` indicates a misspelled word

## Usage

### Installation

```bash
git clone https://github.com/LocalDoc-Azerbaijan/spell_checker_azerbaijani.git
cd spell_checker_azerbaijani
pip install -r requirements.txt
```

### Running the Spell Checker

```python
from spell_checker import load_model_and_tokenizer, predict_word

# Load the pre-trained model and tokenizer
model, tokenizer = load_model_and_tokenizer(
    model_path='best_model.h5', 
    vocab_file='vocab.json', 
    merges_file='merges.txt'
)

# Check a word
word = "abadlaşdırma"
label, prob = predict_word(model, tokenizer, word)
print(f"Word: '{word}' -> {'Correct' if label == 1 else 'Incorrect'}, Confidence: {prob:.4f}")
```

### Interactive Mode

You can also use the interactive mode to check words:

```bash
python spell_checker.py
```

This will prompt you to enter words, which will be classified as either correct or incorrect.


## Requirements

- TensorFlow 2.x
- Hugging Face Tokenizers
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
