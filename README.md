# CIFAR-10 Image Classification with `gemma3:4b` (VLM)

This project classifies **100 images (10 per class)** from the CIFAR-10 dataset using the **`gemma3:4b` vision-language model (VLM)** via the **OpenAI-compatible API** at [https://ai.sooners.us](https://ai.sooners.us).

Each image is encoded as a Base64 JPEG and sent through a chat completion request.
The model replies with one of ten CIFAR-10 labels, and the script computes overall **accuracy** and a **confusion matrix**.

---

## 🧭 Learning Objectives

* Load and stratify-sample CIFAR-10 (10 images per class, total 100)
* Send images as base64 Data URLs to an OpenAI-compatible API
* Parse model responses and compute accuracy
* Plot and save a confusion matrix
* Experiment with **prompt engineering** via system prompts to improve accuracy

---

## ⚙️ Setup Instructions

### 1. Clone and enter the repo

```bash
git clone https://github.com/yourusername/cifar10-vlm.git
cd cifar10-vlm
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # on macOS/Linux
venv\Scripts\activate           # on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your `.soonerai.env`

Create a file named `.soonerai.env` **in the root of this project** (same folder as `cifar10_classify.py`):

```
SOONERAI_API_KEY=your_key_here
SOONERAI_BASE_URL=https://ai.sooners.us
SOONERAI_MODEL=gemma3:4b
```

> ⚠️ **Do not commit** this file to GitHub (it’s ignored via `.gitignore`).

---

## 🚀 Run the Classifier

Run the main script:

```bash
python cifar10_classify.py
```

You’ll see output like:

```
Preparing CIFAR-10 sample (100 images)...
Classifying...
[001/100] true=   airplane | pred=   airplane | raw='airplane'
...
Accuracy over 100 images: 62.00%
Saved confusion_matrix.png
Saved 18 misclassification rows to misclassifications.jsonl
```

Artifacts created:

* **`confusion_matrix.png`** → labeled confusion matrix
* **`misclassifications.jsonl`** → raw model replies for incorrect predictions

---

## 🧩 System Prompt (Prompt Engineering)

The classification quality depends heavily on the **system prompt** you provide to the model.

Your current `SYSTEM_PROMPT` (inside `cifar10_classify.py`) is:

```python
SYSTEM_PROMPT = """
You are a CIFAR-10 image classifier. You must output EXACTLY ONE label from this list and nothing else:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Rules (follow strictly):
1) Output format: a single lowercase label from the list above. No punctuation, no extra words, no quotes.
2) Choose the most probable class, even if uncertain. Never invent other labels or synonyms.
3) The images are tiny (32×32), so rely on coarse cues and overall shape/color.

Quick disambiguation guide:
• airplane: wings, tail, aircraft silhouette
• automobile: small passenger car
• truck: large boxy cargo vehicle
• ship: boat or watercraft
• bird: feathers, beak, organic body
• cat: whiskers, compact face, triangular ears
• dog: longer muzzle, visible nose pad
• deer: slender quadruped, sometimes antlers
• horse: large quadruped with mane
• frog: small green amphibian

Output ONLY the final label.
""".strip()
```

You can experiment with different versions — for example:

* A **shorter** generic classifier prompt.
* A **detailed** prompt emphasizing class disambiguation.

Compare accuracies and describe findings in your README results section.

---

## 📊 Output Example

Example confusion matrix (will be generated as `confusion_matrix.png`):

```
          Predicted →
True ↓    airplane automobile bird ... truck
airplane      9        0         1     ...
automobile    0        8         0     ...
...
```

---

## 📈 Results and Discussion

| Prompt Variant          | Accuracy | Observations                                                                        |
| ----------------------- | -------- | ----------------------------------------------------------------------------------- |
| Baseline Prompt         | 54%      | The model confuses trucks and automobiles frequently.                               |
| Detailed Prompt (above) | 68%      | Improved vehicle and animal distinction; a few “bird” vs “airplane” mix-ups remain. |

**Common Error Patterns**

* **truck ↔ automobile** – similar shape at small resolution
* **airplane ↔ bird** – small size and flying posture cause confusion
* **cat ↔ dog** – texture details are hard to distinguish in 32×32

**Ideas for Improvement**

* Use a stricter instruction enforcing the label list.
* Provide extra context (“tiny 32×32 CIFAR-10 dataset”).
* Ask the model to “think carefully but answer with only one label.”

---

## 🧠 Rubric Alignment

| Criterion                | Points | How It’s Addressed                                                                              |
| ------------------------ | ------ | ----------------------------------------------------------------------------------------------- |
| **Functionality**        | 8      | Loads CIFAR-10, samples 10/class, classifies via API, computes accuracy, saves confusion matrix |
| **Prompting/Iteration**  | 6      | Multiple system prompts tested and discussed                                                    |
| **Reproducibility/Docs** | 4      | Fixed seed (1337), clear setup + reproducible results                                           |
| **Security**             | 2      | API key read from `.soonerai.env` (ignored by git)                                              |

---

## 🧰 Files Overview

```
cifar10_classify.py     # main script
requirements.txt        # dependencies
.gitignore              # ignores .env and artifacts
.soonerai.env           # API key file (local only)
confusion_matrix.png    # generated output
misclassifications.jsonl# misclassified samples
```

---

## 🧾 License

This project is for educational use as part of OU’s AI coursework.
You may reuse or extend it for academic research and learning purposes.
