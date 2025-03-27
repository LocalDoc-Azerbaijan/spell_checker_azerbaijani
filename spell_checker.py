import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tokenizers import ByteLevelBPETokenizer
import argparse


class AttentionLayer(tf.keras.layers.Layer):
    """
    Attention mechanism layer for focusing on the most relevant parts of the input sequence.
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True,
            name='att_weight'
        )
        self.b = self.add_weight(
            shape=(1,),
            initializer='zeros',
            trainable=True,
            name='att_bias'
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return output
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config


def tokenize_text(tokenizer, texts, max_length=30):
    """
    Tokenize and pad text inputs for the model.
    
    Args:
        tokenizer: ByteLevelBPETokenizer instance
        texts: List of strings to tokenize
        max_length: Maximum sequence length for padding
        
    Returns:
        Padded sequences of token IDs
    """
    encoded_texts = []
    for text in texts:
        tokens = tokenizer.encode(text).ids
        encoded_texts.append(tokens)
    padded_texts = pad_sequences(encoded_texts, maxlen=max_length, padding='post')
    return padded_texts


def load_model_and_tokenizer(model_path='best_model.h5', vocab_file='vocab.json', merges_file='merges.txt'):
    """
    Load the trained model and tokenizer.
    
    Args:
        model_path: Path to the saved model file
        vocab_file: Path to the tokenizer vocabulary file
        merges_file: Path to the tokenizer merges file
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model = tf.keras.models.load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
    tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
    return model, tokenizer


def predict_word(model, tokenizer, word, max_length=30):
    """
    Get the raw prediction score for a word.
    
    Args:
        model: Loaded spell-checking model
        tokenizer: ByteLevelBPETokenizer instance
        word: Word to check
        max_length: Maximum sequence length for padding
        
    Returns:
        Raw prediction score (probability)
    """
    input_seq = tokenize_text(tokenizer, [word], max_length)
    prob = model.predict(input_seq, verbose=0)[0][0]
    return prob


def batch_predict(model, tokenizer, words, max_length=30):
    """
    Get raw prediction scores for a batch of words.
    
    Args:
        model: Loaded spell-checking model
        tokenizer: ByteLevelBPETokenizer instance
        words: List of words to check
        max_length: Maximum sequence length for padding
        
    Returns:
        List of dictionaries with prediction results
    """
    input_seq = tokenize_text(tokenizer, words, max_length)
    probs = model.predict(input_seq, verbose=0).flatten()
    
    results = []
    for word, prob in zip(words, probs):
        results.append({
            'word': word,
            'raw_score': float(prob)
        })
    return results


def process_file(model, tokenizer, input_file, output_file, max_length=30):
    """
    Process a file of words and save the raw prediction scores.
    
    Args:
        model: Loaded spell-checking model
        tokenizer: ByteLevelBPETokenizer instance
        input_file: Path to file with one word per line
        output_file: Path to save results
        max_length: Maximum sequence length for padding
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    
    results = batch_predict(model, tokenizer, words, max_length)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("word,raw_score\n")
        for res in results:
            f.write(f"{res['word']},{res['raw_score']:.4f}\n")
    
    print(f"Processed {len(words)} words. Results saved to {output_file}")


def interactive_mode(model, tokenizer, max_length=30):
    """
    Run an interactive spell-checking session.
    """
    print("=== Azerbaijani Spell Checker ===")
    print("Enter a word to get its prediction score (or 'exit' to quit)")
    print("----------------------------------")
    
    while True:
        word = input("\nEnter a word (or 'exit' to quit): ").strip()
        if word.lower() == 'exit':
            break
        
        if not word:
            continue
            
        prob = predict_word(model, tokenizer, word, max_length)
        
        print(f"Word: '{word}'")
        print(f"Raw prediction score: {prob:.4f}")
        print(f"Note: Higher scores suggest incorrect spelling, lower scores suggest correct spelling")


def main():
    parser = argparse.ArgumentParser(description='Azerbaijani Spell Checker')
    parser.add_argument('--model', default='best_model.h5', help='Path to the model file')
    parser.add_argument('--vocab', default='vocab.json', help='Path to the tokenizer vocabulary file')
    parser.add_argument('--merges', default='merges.txt', help='Path to the tokenizer merges file')
    parser.add_argument('--input', help='Input file with one word per line')
    parser.add_argument('--output', help='Output file for batch processing results')
    parser.add_argument('--threshold', type=float, default=None, 
                        help='Optional threshold to apply for classification (0.0-1.0)')
    
    args = parser.parse_args()
    
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.vocab, args.merges)
    print("Model and tokenizer loaded successfully.")
    
    if args.threshold is not None:
        print(f"Using classification threshold: {args.threshold}")
    
    if args.input and args.output:
        process_file(model, tokenizer, args.input, args.output)
    else:
        interactive_mode(model, tokenizer)


if __name__ == '__main__':
    main()