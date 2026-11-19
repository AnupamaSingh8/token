# LLM Pipeline Explorer 🚀

Ek interactive aur educational tool jo LLM pipeline ke starting steps ko visually aur mathematically explore karta hai:
1. **Training Sentences se Vocabulary Build karna** (`file.txt` se).
2. **Out-of-Vocabulary (OOV) Handling**: Jo word training set me nahi hai uska Token ID strictly `0` (`<UNK>`) hona.
3. **Sentence Input & Tokenizer**: Kisi bhi custom sentence ka tokens aur token IDs nikalna.
4. **Embedding Selection**: Spelling-based (character) vs Meaning-based (semantic) embeddings ka selection.
5. **Interactive 2D Graph**: Embedding space ka 2D visual scatter graph jisme sentence sequence flow aur clusters dikhte hain.

---

## 📂 Project Structure

- `file.txt` - Aapke training sentences (7 sentences).
- `pipeline_explorer.py` - Core Python engine (Vocabulary builder, Tokenizer, 2D SVD/PCA Spelling projector, Semantic clusters, aur terminal ASCII graph).
- `app.py` - Flask web server with REST APIs (`/api/vocab`, `/api/tokenize`).
- `index.html` - Standalone interactive browser app (HTML5 + Tailwind + Canvas).

---

## ⚡ Kaise Chalayein (How to Run)

### Option 1: Direct Browser me Open karein (Instant)
Aap bina kisi server ke direct browser me dekh sakte hain:
```bash
open index.html
```

### Option 2: Python Web Server (Flask)
```bash
python3 app.py
```
Phir browser me open karein: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Option 3: Terminal / CLI Mode
Agar terminal me hi dekhna hai:
```bash
python3 pipeline_explorer.py
```

---

## 🧠 Core Concepts & LLM Pipeline Steps

### Step 1: Vocabulary Building (`file.txt`)
Training file se 7 sentences padhe jaate hain:
1. *The cat sat on the mat.*
2. *The dog chased the cat.*
3. *Birds fly high in the sky.*
4. *Fish swim in the clear blue river.*
5. *The cat sat on the wooden chair.*
6. *I love Programming.*
7. *Coding in python is my passion.*

Isme se total 27 unique words nikale gaye aur order of appearance ke hisab se IDs assign kiye gaye:
- `ID: 0` ➔ `<UNK>` (Unknown / Out-of-Vocabulary token)
- `ID: 1` ➔ `the`
- `ID: 2` ➔ `cat`
- `ID: 3` ➔ `sat`
...
- `ID: 27` ➔ `passion`

### Step 2: Tokenizer with OOV (ID = 0)
Jab aap koi input sentence dete hain jaise:
> `"The cat loves coding in python"`

- `the` ➔ ID: 1 (In-Vocab)
- `cat` ➔ ID: 2 (In-Vocab)
- `loves` ➔ **ID: 0 (<UNK>)** *(Kyunki training data me 'love' tha, 'loves' nahi!)*
- `coding` ➔ ID: 23 (In-Vocab)
- `in` ➔ ID: 11 (In-Vocab)
- `python` ➔ ID: 24 (In-Vocab)

### Step 3: Spelling-based vs Meaning-based Embeddings
| Feature | 🔤 Spelling-Based Embedding | 🧠 Meaning-Based (Semantic) Embedding |
|---|---|---|
| **Basis** | Character n-grams / letter spelling | Conceptual context / meaning / domain |
| **`cat` vs `mat`** | **Bahut Paas** (dono me `-at` hai) | **Dur** (cat animal hai, mat inanimate object) |
| **`cat` vs `dog`** | **Dur** (letters bilkul alag hain) | **Bahut Paas** (dono domestic pets/animals hain) |
| **`coding` vs `programming`** | Dur (letters match nahi karte) | **Bahut Paas** (dono software development hain) |
| **LLM Context** | Subword tokenizers (e.g. BPE, WordPiece) | Dense embeddings (Word2Vec, Transformer layers) |

### Step 4: 2D Graph Visualizer
- Saare 27 vocabulary words background me coordinate space me plotted hain.
- Aapka input sentence glowing markers aur sequence numbers (`#1, #2, #3...`) se highlight hota hai.
- Ek dashed line sentence ke tokens ka path / trajectory draw karti hai.
- Hover karne par Token ID, Word, aur (X, Y) coordinates milte hain.
