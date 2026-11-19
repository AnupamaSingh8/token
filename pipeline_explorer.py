"""
LLM Pipeline Explorer
=====================
Part 1: Vocabulary Builder (from file.txt)
Part 2: Tokenizer with OOV (<UNK> = 0)
Part 3: Embeddings (Spelling-based vs Meaning-based)
Part 4: 2D Graph Visualizer
"""

import os
import re
import math
import random
import json

class LLMPipeline:
    def __init__(self, training_file="file.txt"):
        self.training_file = training_file
        self.sentences = []
        self.vocab = {}          # word -> token_id
        self.id_to_word = {}     # token_id -> word
        self.char_bigrams = []
        self.bg_idx = {}
        self.spelling_coords = {}
        self.semantic_coords = {}
        self.semantic_categories = {}
        
        self.load_training_data()
        self.build_vocabulary()
        self.init_spelling_embeddings()
        self.init_semantic_embeddings()

    def load_training_data(self):
        """Read training sentences from file.txt"""
        if not os.path.exists(self.training_file):
            raise FileNotFoundError(f"Training file '{self.training_file}' not found.")
        
        with open(self.training_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Filter out headers like 'Part 1: Tokenizer', 'Training Sentences'
        self.sentences = [
            line for line in lines 
            if not line.lower().startswith("part") 
            and not line.lower().startswith("training")
        ]

    def build_vocabulary(self):
        """
        Build vocabulary from training sentences.
        Unknown / OOV token is assigned ID 0 (<UNK>).
        Vocabulary words get IDs starting from 1 in order of appearance.
        """
        self.vocab = {"<UNK>": 0}
        self.id_to_word = {0: "<UNK>"}
        
        for sentence in self.sentences:
            words = re.findall(r"\b[a-zA-Z0-9_\'-]+\b", sentence.lower())
            for word in words:
                if word not in self.vocab:
                    new_id = len(self.vocab)
                    self.vocab[word] = new_id
                    self.id_to_word[new_id] = word

    def tokenize(self, sentence):
        """
        Tokenize an input sentence into tokens and token IDs.
        If a token is not in the vocabulary, its ID is 0 (<UNK>).
        """
        raw_tokens = re.findall(r"\b[a-zA-Z0-9_\'-]+\b", sentence.lower())
        tokens_info = []
        for tok in raw_tokens:
            token_id = self.vocab.get(tok, 0)
            is_oov = (token_id == 0)
            tokens_info.append({
                "token": tok,
                "token_id": token_id,
                "is_oov": is_oov,
                "display": tok if not is_oov else f"{tok} (<UNK>)"
            })
        return tokens_info

    def _get_bigrams(self, word):
        w = f"^{word}$"
        return [w[i:i+2] for i in range(len(w) - 1)]

    def init_spelling_embeddings(self):
        """
        Character n-gram based orthographic embedding projected to 2D coordinates.
        Words with similar letters (e.g. cat, mat, sat) cluster together.
        """
        vocab_words = [w for w in self.vocab.keys() if w != "<UNK>"]
        all_bg = sorted(list(set(bg for w in vocab_words for bg in self._get_bigrams(w))))
        self.char_bigrams = all_bg
        self.bg_idx = {bg: i for i, bg in enumerate(all_bg)}
        dim = len(all_bg)

        vecs = {}
        for w in vocab_words:
            v = [0.0] * dim
            for bg in self._get_bigrams(w):
                if bg in self.bg_idx:
                    v[self.bg_idx[bg]] += 1.0
            norm = math.sqrt(sum(x * x for x in v))
            vecs[w] = [x / norm if norm > 0 else 0.0 for x in v]

        self.spelling_mean = [sum(vecs[w][j] for w in vocab_words) / len(vocab_words) for j in range(dim)]
        centered = {w: [vecs[w][j] - self.spelling_mean[j] for j in range(dim)] for w in vocab_words}

        # Power iteration to find 2 principal components
        random.seed(42)
        def get_pc(prev=[]):
            v = [random.uniform(-1, 1) for _ in range(dim)]
            for p in prev:
                dot = sum(v[i] * p[i] for i in range(dim))
                v = [v[i] - dot * p[i] for i in range(dim)]
            norm = math.sqrt(sum(x * x for x in v))
            v = [x / norm for x in v]
            for _ in range(80):
                projs = {w: sum(centered[w][j] * v[j] for j in range(dim)) for w in vocab_words}
                v_next = [0.0] * dim
                for w in vocab_words:
                    for j in range(dim):
                        v_next[j] += centered[w][j] * projs[w]
                for p in prev:
                    dot = sum(v_next[i] * p[i] for i in range(dim))
                    v_next = [v_next[i] - dot * p[i] for i in range(dim)]
                norm = math.sqrt(sum(x * x for x in v_next))
                if norm < 1e-9:
                    break
                v = [x / norm for x in v_next]
            return v

        self.spelling_pc1 = get_pc([])
        self.spelling_pc2 = get_pc([self.spelling_pc1])

        raw_coords = {
            w: (
                sum(centered[w][j] * self.spelling_pc1[j] for j in range(dim)),
                sum(centered[w][j] * self.spelling_pc2[j] for j in range(dim))
            ) for w in vocab_words
        }
        max_val = max(max(abs(pt[0]), abs(pt[1])) for pt in raw_coords.values())
        self.spelling_scale = 3.2 / max_val if max_val > 0 else 1.0

        self.spelling_coords = {
            w: (round(raw_coords[w][0] * self.spelling_scale, 2),
                round(raw_coords[w][1] * self.spelling_scale, 2))
            for w in vocab_words
        }
        self.spelling_coords["<UNK>"] = (0.0, 0.0)

    def project_spelling_word(self, word):
        """Project any word (even unknown/OOV) into 2D spelling space"""
        dim = len(self.char_bigrams)
        v = [0.0] * dim
        for bg in self._get_bigrams(word):
            if bg in self.bg_idx:
                v[self.bg_idx[bg]] += 1.0
        norm = math.sqrt(sum(x * x for x in v))
        if norm > 0:
            v = [x / norm for x in v]
        c_v = [v[j] - self.spelling_mean[j] for j in range(dim)]
        x = sum(c_v[j] * self.spelling_pc1[j] for j in range(dim)) * self.spelling_scale
        y = sum(c_v[j] * self.spelling_pc2[j] for j in range(dim)) * self.spelling_scale
        return (round(x, 2), round(y, 2))

    def init_semantic_embeddings(self):
        """
        Meaning-based semantic embeddings grouped by conceptual domain.
        Words with similar meanings (e.g. cat & dog, coding & programming) cluster together.
        """
        semantic_groups = {
            "Animals": {
                "center": (-2.8, 2.3),
                "words": ["cat", "dog", "birds", "fish"],
                "color": "#3b82f6"
            },
            "Actions / Verbs": {
                "center": (-0.8, 1.4),
                "words": ["sat", "chased", "fly", "swim"],
                "color": "#8b5cf6"
            },
            "Tech / Passion": {
                "center": (3.0, 2.2),
                "words": ["coding", "programming", "python", "love", "passion"],
                "color": "#10b981"
            },
            "Nature / Environment": {
                "center": (-2.4, -2.4),
                "words": ["sky", "river", "blue", "clear", "high"],
                "color": "#06b6d4"
            },
            "Furniture / Objects": {
                "center": (0.6, -2.6),
                "words": ["mat", "chair", "wooden"],
                "color": "#f59e0b"
            },
            "Grammar / Connectors": {
                "center": (2.6, -1.4),
                "words": ["the", "in", "on", "is", "i", "my"],
                "color": "#ec4899"
            }
        }

        self.semantic_coords = {}
        for cat_name, data in semantic_groups.items():
            cx, cy = data["center"]
            words = data["words"]
            n = len(words)
            for i, w in enumerate(words):
                self.semantic_categories[w] = cat_name
                angle = (2 * math.pi * i) / n
                radius = 0.55 if n > 1 else 0.0
                self.semantic_coords[w] = (
                    round(cx + radius * math.cos(angle), 2),
                    round(cy + radius * math.sin(angle), 2)
                )

        self.semantic_coords["<UNK>"] = (0.0, 0.0)
        self.semantic_categories["<UNK>"] = "Unknown"

    def get_token_embedding(self, token, mode="meaning"):
        """Get 2D coordinates for a token under chosen embedding mode"""
        token = token.lower()
        if mode == "spelling":
            if token in self.spelling_coords:
                return self.spelling_coords[token]
            return self.project_spelling_word(token)
        else: # meaning
            if token in self.semantic_coords:
                return self.semantic_coords[token]
            return (0.0, 0.0)

    def print_summary(self):
        """Print vocabulary summary"""
        print("=" * 60)
        print("LLM PIPELINE EXPLORER: VOCABULARY & TOKENIZER")
        print("=" * 60)
        print(f"Loaded {len(self.sentences)} training sentences from {self.training_file}:")
        for i, s in enumerate(self.sentences, 1):
            print(f"  {i}. {s}")
        print("\nVocabulary Size:", len(self.vocab))
        print("Special Token: <UNK> -> ID: 0 (Assigned to all unseen words)")
        print("\nVocabulary Table:")
        print(f"{'Token ID':<10} {'Word':<15} {'Category':<22}")
        print("-" * 50)
        for w, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
            cat = self.semantic_categories.get(w, "Special / Out of Vocab")
            print(f"{idx:<10} {w:<15} {cat:<22}")
        print("=" * 60)

    def print_interactive_tokens(self, sentence, mode="meaning"):
        """Analyze a sentence and display tokens, IDs, and embedding coordinates"""
        tokens = self.tokenize(sentence)
        print(f"\n--- Input Sentence: \"{sentence}\" ---")
        print(f"Embedding Mode: {mode.upper()}-BASED\n")
        print(f"{'Step':<6} {'Token':<14} {'Token ID':<10} {'Status':<16} {'Coords (X, Y)':<18}")
        print("-" * 66)
        for i, t in enumerate(tokens, 1):
            coords = self.get_token_embedding(t["token"], mode=mode)
            status = "IN-VOCAB" if not t["is_oov"] else "<UNK> (OOV=0)"
            print(f"{i:<6} {t['token']:<14} {t['token_id']:<10} {status:<16} {str(coords):<18}")

    def render_ascii_graph(self, sentence, mode="meaning", width=60, height=22):
        """Render a clean 2D ASCII vector space graph in the terminal"""
        tokens = self.tokenize(sentence)
        input_words = [t["token"] for t in tokens]
        
        # Grid bounds
        min_x, max_x = -3.8, 3.8
        min_y, max_y = -3.5, 3.5

        def to_grid(x, y):
            gx = int((x - min_x) / (max_x - min_x) * (width - 1))
            gy = int((max_y - y) / (max_y - min_y) * (height - 1))
            return max(0, min(width - 1, gx)), max(0, min(height - 1, gy))

        grid = [[" " for _ in range(width)] for _ in range(height)]

        # Draw axes
        ox, oy = to_grid(0, 0)
        for r in range(height):
            grid[r][ox] = "│"
        for c in range(width):
            grid[oy][c] = "─"
        grid[oy][ox] = "┼"

        # Plot all vocab words
        coords_dict = self.spelling_coords if mode == "spelling" else self.semantic_coords
        for w, (x, y) in coords_dict.items():
            if w == "<UNK>": continue
            gx, gy = to_grid(x, y)
            grid[gy][gx] = "·"

        # Plot input words with sequence numbers
        token_points = []
        for seq_idx, t in enumerate(tokens, 1):
            w = t["token"]
            coords = self.get_token_embedding(w, mode=mode)
            gx, gy = to_grid(coords[0], coords[1])
            symbol = str(seq_idx) if seq_idx < 10 else "*"
            grid[gy][gx] = symbol
            token_points.append((gx, gy, w, t["token_id"], seq_idx))

        print(f"\n2D EMBEDDING SPACE GRAPH [{mode.upper()} MODE]:")
        print("+" + "-" * width + "+")
        for row in grid:
            print("|" + "".join(row) + "|")
        print("+" + "-" * width + "+")
        print("Legend: '·' = Vocab word, '1, 2, ...' = Input sentence sequence")
        for gx, gy, w, tid, s_idx in token_points:
            coords = self.get_token_embedding(w, mode=mode)
            print(f"  [{s_idx}] {w:<12} (Token ID: {tid}) -> Coords: {coords}")


if __name__ == "__main__":
    pipeline = LLMPipeline("file.txt")
    pipeline.print_summary()
    
    test_sentence = "The cat sat on the mat."
    pipeline.print_interactive_tokens(test_sentence, mode="meaning")
    pipeline.render_ascii_graph(test_sentence, mode="meaning")

    print("\n" + "=" * 60)
    print("Testing Spelling-based vs Meaning-based differences:")
    print("In Spelling mode: 'cat' and 'mat' are close (same letters)")
    print("In Meaning mode: 'cat' and 'dog' are close (both animals)")
    test_sentence_2 = "The cat and dog loved coding in python."
    pipeline.print_interactive_tokens(test_sentence_2, mode="spelling")
    pipeline.render_ascii_graph(test_sentence_2, mode="spelling")
