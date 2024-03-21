import torch.optim as optim
import time
import os
import streamlit as st
import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt # for making figures
from pprint import pprint
import re
import random
import string

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Preprocess the text
    # Remove unnecessary characters, such as tabs and extra whitespaces
    text = text.replace('\t', ' ')
    text = ' '.join(text.split())

    # Tokenize whitespace and newline characters
    text = text.replace('\n', '')
    
    return text

# Load and preprocess data
file_path = r'C:\Users\DELL\OneDrive\Desktop\Paul.txt'
words = preprocess_text_file(file_path)
# Create vocabulary of unique characters
vocab = sorted(list(set(''.join(words))))

# Add special tokens to vocabulary
stoi = {s:i+1 for i,s in enumerate(vocab)}

stoi['_'] = 0

# Build the vocabulary of characters and mappings to/from integers
itos = {i: s for s, i in stoi.items()} # pprint(itos)

block_size = 5  # context length: how many characters do we take to predict the next one?
X, Y = [], []
context = [0] * block_size
for w in words[:]:
        
    for ch in w:
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        # print(''.join(itos[i] for i in context), '--->', itos[ix])
        context = context[1:] + [ix]  # Crop and append
    
# Move data to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = torch.tensor(X).to(device)
Y = torch.tensor(Y).to(device)

# Define model architecture
class NextChar(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x

def train_model(model, X, Y, loss_fn, opt, epochs, batch_size, print_every=100):
    # device = model.device
    # model.train()
    for epoch in range(epochs):
        start_time = time.time()
        for i in range(0, X.shape[0], batch_size):
            x = X[i:i+batch_size]
            y = Y[i:i+batch_size]
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
        end_time = time.time()
        if epoch % print_every == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}, Time: {end_time - start_time}")

def train_and_save_models(min_emb, max_emb):
    block_size = 5  # context length: how many characters do we take to predict the next one?
    batch_size = 4096
    epochs = 101
    lr = 0.01
    print_every = 100

    # Train and save models
    for emb_dim in range(min_emb, max_emb + 1):
        model = NextChar(block_size, len(stoi), emb_dim, 10).to(device)
        loss_fn = nn.CrossEntropyLoss()
        opt = optim.AdamW(model.parameters(), lr=lr)
        
        train_model(model, X, Y, loss_fn, opt, epochs, batch_size, print_every)

        # Save model
        model_path = f"model_emb{emb_dim}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model with embedding size {emb_dim} saved to {model_path}")

def load_model(emb_dim):
    model = NextChar(block_size, len(stoi), emb_dim, 10).to(device)
    model_path = f"model_emb{emb_dim}.pt"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def generate_text_with_model(model, input_text, itos, stoi, block_size, length=100):
    context = [0] * block_size
    input_indices = [stoi[ch] for ch in input_text if ch in stoi]
    if len(input_indices) > block_size:
        context = input_indices[-block_size:]
    else:
        context[block_size - len(input_indices):] = input_indices
    generated_text = input_text
    with torch.no_grad():
        for i in range(length):
            x = torch.tensor(context).view(1, -1).to(device)
            y_pred = model(x)
            ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()

            next_char = itos[ix]
            if next_char == '_':
                generated_text += ''
            else:
                generated_text += next_char
            context = context[1:] + [ix]
    return generated_text

# Streamlit app layout and functionality
import streamlit as st

def main():
    st.title("Text Generation with PyTorch and Streamlit")
    st.sidebar.header("Text Generation Settings")
    block_size = 5
    input_text = st.sidebar.text_input("Input Text", value="April 1995")
    length = st.sidebar.slider("Length of Generated Text", min_value=10, max_value=1000, value=500, step=10)
    min_emb = 2
    max_emb = 5
    emb_dim = st.sidebar.slider("Select Embedding Size", min_value=min_emb, max_value=max_emb, value=min_emb, step=1)
    if st.sidebar.button("Train and Save Models"):
        train_and_save_models(min_emb, max_emb)
        st.sidebar.success("Models trained and saved successfully!")

    if st.sidebar.button("Generate Text"):
        model = load_model(emb_dim)
        generated_text = generate_text_with_model(model, input_text, itos, stoi, block_size, length)
        st.subheader("Generated Text:")
        st.write(generated_text)

if __name__ == "__main__":
    main()
