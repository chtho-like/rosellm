import json
import math
import os
import re
from collections import Counter

import nltk
import regex
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BytePairEncoder:
    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size
        self.encoder = {}  # token -> id
        self.decoder = {}  # id -> token
        self.bpe_ranks = {}
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.pat = regex.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.special_tokens = {self.pad_token: 0, self.eos_token: 1}

    def _bytes_to_unicode(self):
        """
        Returns a mapping from bytes to unicode strings.
        This handles the ASCII bytes that don't need remapping and creates
        a mapping for bytes that aren't printable ASCII.
        """
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    def _get_pairs(self, word):
        """
        Return set of symbol pairs in a word.
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def train(self, texts, min_frequency=2, verbose=True):
        """
        Train a BPE tokenizer on the given texts.
        """
        # Count word frequencies
        word_freqs = Counter()
        for text in tqdm(texts, desc="Counting words", disable=not verbose):
            words = regex.findall(self.pat, text)
            for word in words:
                word_encoded = "".join(
                    self.byte_encoder[b] for b in word.encode("utf-8")
                )
                word_freqs[word_encoded] += 1

        # Filter by frequency
        word_freqs = {
            word: freq for word, freq in word_freqs.items() if freq >= min_frequency
        }

        # Initialize vocabulary with characters
        vocab = set()
        for word in word_freqs:
            for char in word:
                vocab.add(char)

        # Add special tokens to vocabulary
        vocab = list(self.special_tokens.keys()) + list(vocab)

        # Initialize merge operations
        merges = []

        # Iteratively merge most frequent pairs
        target_vocab_size = min(
            self.vocab_size, len(vocab) + 2000
        )  # Limit iterations to prevent hanging
        pbar = tqdm(
            total=target_vocab_size - len(vocab),
            desc="Training BPE",
            disable=not verbose,
        )

        max_iterations = 10000  # Safety limit
        iteration_count = 0

        while len(vocab) < self.vocab_size and iteration_count < max_iterations:
            iteration_count += 1

            # Count pair frequencies
            pair_freqs = Counter()
            for word, freq in word_freqs.items():
                pairs = self._get_pairs(word)
                if not pairs:  # Skip words without pairs
                    continue
                for pair in pairs:
                    pair_freqs[pair] += freq

            if not pair_freqs:
                print("No more pairs to merge, stopping early.")
                break

            # Find most frequent pair
            best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]

            # Create new token by merging the pair
            new_token = "".join(best_pair)
            vocab.append(new_token)
            merges.append(best_pair)

            # Update words with the merged pair
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = word
                i = new_word.find(best_pair[0] + best_pair[1])
                if i != -1:  # Only process words that contain the pair
                    new_word = new_word.replace(best_pair[0] + best_pair[1], new_token)
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs

            pbar.update(1)

            if len(vocab) >= self.vocab_size:
                break

        pbar.close()

        if iteration_count >= max_iterations:
            print(f"Stopped after {max_iterations} iterations to prevent hanging.")

        # Create encoder and decoder
        for i, token in enumerate(vocab):
            if token in self.special_tokens:
                self.encoder[token] = self.special_tokens[token]
            else:
                self.encoder[token] = (
                    len(self.special_tokens) + i - len(self.special_tokens)
                )

        self.decoder = {v: k for k, v in self.encoder.items()}

        # Create BPE ranks
        self.bpe_ranks = dict(zip(merges, range(len(merges))))

        self.pad_token_id = self.encoder[self.pad_token]
        self.eos_token_id = self.encoder[self.eos_token]

        return self

    def bpe(self, token):
        """
        Apply BPE encoding to a token.
        """
        word = tuple(token)
        pairs = self._get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0

            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break

            pairs = self._get_pairs(word)

        return "".join(word)

    def tokenize(self, text):
        """
        Tokenize text using BPE.
        """
        tokens = []
        for token in regex.findall(self.pat, text):
            token_encoded = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens = self.bpe(token_encoded).split(" ")
            tokens.extend(bpe_tokens)
        return tokens

    def convert_tokens_to_ids(self, tokens):
        """
        Convert tokens to ids.
        """
        return [
            self.encoder.get(token, self.encoder.get(self.pad_token))
            for token in tokens
        ]

    def __call__(
        self,
        texts,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors=None,
    ):
        batch_input_ids = []
        batch_attention_mask = []

        for text in texts:
            tokens = self.tokenize(text)
            if truncation and len(tokens) > max_length - 1:  # -1 for EOS token
                tokens = tokens[: max_length - 1]

            # Add EOS token
            tokens.append(self.eos_token)

            # Convert to ids
            input_ids = self.convert_tokens_to_ids(tokens)

            # Pad to max_length if needed
            if padding == "max_length":
                attention_mask = [1] * len(input_ids)
                padding_length = max_length - len(input_ids)
                input_ids = input_ids + [self.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)

        # Convert to tensors if requested
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            }
        return {"input_ids": batch_input_ids, "attention_mask": batch_attention_mask}

    def save(self, path):
        """
        Save the tokenizer to a directory.
        """
        os.makedirs(path, exist_ok=True)

        # Save encoder, decoder, and bpe_ranks
        with open(os.path.join(path, "encoder.json"), "w") as f:
            json.dump(self.encoder, f)

        with open(os.path.join(path, "vocab.bpe"), "w") as f:
            for pair, rank in sorted(self.bpe_ranks.items(), key=lambda x: x[1]):
                f.write(f"{pair[0]} {pair[1]}\n")

    @classmethod
    def from_pretrained(cls, path):
        """
        Load a tokenizer from a directory.
        """
        tokenizer = cls()

        # Load encoder
        with open(os.path.join(path, "encoder.json"), "r") as f:
            tokenizer.encoder = json.load(f)

        # Load decoder
        tokenizer.decoder = {v: k for k, v in tokenizer.encoder.items()}

        # Load bpe_ranks
        tokenizer.bpe_ranks = {}
        with open(os.path.join(path, "vocab.bpe"), "r") as f:
            for i, line in enumerate(f):
                first, second = line.strip().split()
                tokenizer.bpe_ranks[(first, second)] = i

        tokenizer.pad_token_id = tokenizer.encoder[tokenizer.pad_token]
        tokenizer.eos_token_id = tokenizer.encoder[tokenizer.eos_token]

        return tokenizer


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.tokenize(text)
        # Truncate if needed
        if len(tokens) > self.max_length - 1:  # -1 for EOS token
            tokens = tokens[: self.max_length - 1]

        # Add EOS token
        tokens.append(self.tokenizer.eos_token)

        # Convert to IDs and check they're within vocabulary size
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)

        # Pad sequences to max_length
        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length

        # Verify all IDs are valid
        vocab_size = len(self.tokenizer.encoder)
        input_ids = [
            min(id, vocab_size - 1) for id in input_ids
        ]  # Clip to prevent out-of-bounds

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # Labels are the same as inputs for language modeling
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self):
        return len(self.texts)


class GPT2Attention(nn.Module):
    def __init__(self, nx, n_head, scale=False):
        super().__init__()
        assert nx % n_head == 0, "nx must be divisible by n_head"
        self.n_head = n_head
        self.split_size = nx
        self.scale = scale

        # Ensure head size is compatible with embedding dimension
        self.head_dim = nx // n_head

        self.c_attn = nn.Linear(nx, 3 * nx)
        self.c_proj = nn.Linear(nx, nx)

        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)

    def _attn(self, q, k, v, attention_mask=None):
        # Scale dot-product attention
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(self.head_dim)

        # Apply the attention mask
        if attention_mask is not None:
            # Convert attention mask from 0/1 to -10000.0/0.0
            attention_mask = (1.0 - attention_mask) * -10000.0
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        outputs = torch.matmul(w, v)
        return outputs

    def merge_heads(self, x):
        """
        Merge attention heads
        Input shape: [batch_size, n_head, seq_length, head_dim]
        Output shape: [batch_size, seq_length, n_head * head_dim]
        """
        batch_size, n_head, seq_length, head_dim = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_length, n_head * head_dim)

    def split_heads(self, x):
        """
        Split embeddings into attention heads
        Input shape: [batch_size, seq_length, n_head * head_dim]
        Output shape: [batch_size, n_head, seq_length, head_dim] for query/value
                     [batch_size, n_head, head_dim, seq_length] for key
        """
        batch_size, seq_length, _ = x.size()
        x = x.view(batch_size, seq_length, self.n_head, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, attention_mask=None):
        batch_size, seq_length, _ = x.size()

        # Apply linear layer and split into query, key, value
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)

        # Split heads
        query = self.split_heads(query)  # [batch, n_head, seq_length, head_dim]
        key = self.split_heads(key)  # [batch, n_head, seq_length, head_dim]
        value = self.split_heads(value)  # [batch, n_head, seq_length, head_dim]

        # Transpose key for dot product with query
        key = key.transpose(-1, -2)  # [batch, n_head, head_dim, seq_length]

        # Apply attention
        attn_outputs = self._attn(query, key, value, attention_mask)

        # Merge heads
        attn_outputs = self.merge_heads(attn_outputs)

        # Apply output projection
        attn_outputs = self.c_proj(attn_outputs)
        attn_outputs = self.resid_dropout(attn_outputs)

        return attn_outputs


class GPT2MLP(nn.Module):
    def __init__(self, n_state, nx):
        super().__init__()
        self.c_fc = nn.Linear(nx, n_state)
        self.c_proj = nn.Linear(n_state, nx)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class GPT2Block(nn.Module):
    def __init__(self, nx, n_head, scale=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(nx)
        self.attn = GPT2Attention(nx, n_head, scale)
        self.ln_2 = nn.LayerNorm(nx)
        self.mlp = GPT2MLP(4 * nx, nx)

    def forward(self, x, attention_mask=None):
        a = self.attn(self.ln_1(x), attention_mask)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x


class GPT2Model(nn.Module):
    def __init__(self, vocab_size, n_layer=12, n_head=12, n_embd=768):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(1024, n_embd)  # Position embeddings
        self.drop = nn.Dropout(0.1)
        self.h = nn.ModuleList(
            [GPT2Block(n_embd, n_head, scale=True) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, input_ids, attention_mask=None):
        device = input_ids.device

        # Get input shape
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]

        # Create position ids
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Get embeddings
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        # Prepare attention mask
        extended_attention_mask = None
        if attention_mask is not None:
            # Convert from [batch, seq_len] to [batch, 1, 1, seq_len]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Apply transformer blocks
        for block in self.h:
            hidden_states = block(hidden_states, extended_attention_mask)

        hidden_states = self.ln_f(hidden_states)

        return hidden_states


class GPT2LMHeadModel(nn.Module):
    def __init__(self, vocab_size, n_layer=12, n_head=12, n_embd=768):
        super().__init__()
        self.transformer = GPT2Model(vocab_size, n_layer, n_head, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Tie weights between embedding and output layer
        self.lm_head.weight = self.transformer.wte.weight

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def generate(self, input_ids, max_length=50, temperature=1.0, attention_mask=None):
        """
        Generate text using the model.
        """
        for _ in range(max_length):
            # Get the model's output for the current input
            with torch.no_grad():
                transformer_outputs = self.transformer(input_ids, attention_mask)
                logits = self.lm_head(transformer_outputs)

                # Get the logits for the last token
                next_token_logits = logits[:, -1, :] / temperature

                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append the new token to the input
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # If EOS token is generated, stop
                if next_token.item() == 1:  # EOS token ID
                    break

        return input_ids

    def forward(self, input_ids, attention_mask=None, labels=None):
        transformer_outputs = self.transformer(input_ids, attention_mask)
        lm_logits = self.lm_head(transformer_outputs)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        return type("ModelOutput", (), {"loss": loss, "logits": lm_logits})()


def main():
    # Download and use a real dataset - Tiny Shakespeare
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    # Also download punkt_tab which is needed for sent_tokenize
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")

    # Download Tiny Shakespeare dataset
    shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(shakespeare_url)
    shakespeare_text = response.text

    # Split into sentences for easier processing
    sentences = sent_tokenize(shakespeare_text)
    print(f"Downloaded {len(sentences)} sentences from Tiny Shakespeare dataset")

    # Take a subset for faster training in this example
    texts = sentences[:1000]  # Use first 1000 sentences

    # Create and train tokenizer with smaller vocab to prevent issues
    vocab_size = 1000  # Reduced vocab size for safer training
    tokenizer = BytePairEncoder(vocab_size=vocab_size)
    tokenizer.train(texts, verbose=True)

    # Save tokenizer for future use
    tokenizer.save("./tokenizer")

    # Create a very small model for demonstration purposes
    n_embd = 64  # Smaller embedding dimension
    n_head = 4  # Number of attention heads

    # Ensure embedding dimension is divisible by the number of heads
    assert (
        n_embd % n_head == 0
    ), "Embedding dimension must be divisible by the number of heads"

    # Use CPU for better error messages during development
    safe_device = "cpu"  # Can change back to device after fixing issues

    model = GPT2LMHeadModel(
        vocab_size=vocab_size,
        n_layer=2,  # Even smaller for testing
        n_head=n_head,
        n_embd=n_embd,
    ).to(safe_device)

    # Create dataset and dataloader
    dataset = TextDataset(texts, tokenizer)
    batch_size = 2  # Smaller batch for testing
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer with smaller learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Training loop
    num_epochs = 1  # Start with just one epoch for testing
    print(f"Starting training with vocab_size={vocab_size}, batch_size={batch_size}")

    try:
        for e in range(num_epochs):
            total_loss = 0
            for batch, data in enumerate(data_loader):
                # Move data to device
                input_ids = data["input_ids"].to(safe_device)
                attention_mask = data["attention_mask"].to(safe_device)
                labels = data["labels"].to(safe_device)

                # Verify input_ids are within bounds
                if torch.max(input_ids) >= vocab_size:
                    print(
                        f"Warning: Found token ID {torch.max(input_ids).item()} >= vocab_size {vocab_size}"
                    )
                    input_ids = torch.clamp(input_ids, max=vocab_size - 1)

                # Forward pass
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch % 10 == 0:
                    print(
                        f"Epoch {e+1}/{num_epochs}, Batch {batch}/{len(data_loader)}: Loss {loss.item():.4f}"
                    )

            avg_loss = total_loss / len(data_loader)
            print(f"Epoch {e+1}/{num_epochs} completed. Average loss: {avg_loss:.4f}")

    except Exception as e:
        print(f"Error during training: {str(e)}")

    # Save the trained model
    try:
        torch.save(model.state_dict(), "tiny_shakespeare_gpt2.pt")
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

    # Generate some text as a demonstration
    try:
        model.eval()
        with torch.no_grad():
            # Start with a prompt
            prompt = "To be or not to be"

            # Tokenize the prompt directly
            tokens = tokenizer.tokenize(prompt)
            # convert_tokens_to_ids may return Optional[int]; coerce Nones to 0 for safety
            from typing import List, Optional

            input_ids_opt: List[Optional[int]] = tokenizer.convert_tokens_to_ids(tokens)
            input_ids_list: List[int] = [int(x) if x is not None else 0 for x in input_ids_opt]

            # Check for out-of-bounds tokens
            if max(input_ids_list) >= vocab_size:
                print(
                    f"Warning: Prompt contains token ID >= vocab_size. Clipping values."
                )
                input_ids_list = [min(tok_id, vocab_size - 1) for tok_id in input_ids_list]

            input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(safe_device)

            # Generate text
            try:
                # Generate text by autoregressively predicting one token at a time
                max_length = 20  # Generate fewer tokens for testing

                # Manual generation instead of using model.generate
                for _ in range(max_length):
                    # Forward pass
                    outputs = model(input_ids)
                    next_token_logits = outputs.logits[:, -1, :]

                    # Get the most likely next token
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

                    # Append to input_ids
                    input_ids = torch.cat([input_ids, next_token], dim=-1)

                    # Stop if EOS token is generated
                    if next_token.item() == tokenizer.eos_token_id:
                        break

                # Convert generated IDs to text
                print(f"Generated text from prompt '{prompt}':")
                for token_id in input_ids[0]:
                    token_id_int = token_id.item()
                    if token_id_int in tokenizer.decoder:
                        print(tokenizer.decoder[token_id_int], end="")
            except Exception as e:
                print(f"Error during generation: {str(e)}")
    except Exception as e:
        print(f"Error in text generation section: {str(e)}")


if __name__ == "__main__":
    main()
