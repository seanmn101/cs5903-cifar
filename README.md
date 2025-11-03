# CIFAR-10 Image Classification

This project classifies **100 images (10 per class)** from the CIFAR-10 dataset using the **`gemma3:4b` vision-language model (VLM)** via the **OpenAI-compatible API** at [https://ai.sooners.us](https://ai.sooners.us).

Each image is encoded as a Base64 JPEG and sent through a chat completion request.
The model replies with one of ten CIFAR-10 labels, and the script computes overall **accuracy** and a **confusion matrix**.

---

### 1. Clone and enter the repo

```bash
git clone https://github.com/yourusername/cifar10-vlm.git
cd cifar10-vlm
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your `.soonerai.env`

Create a file named `.soonerai.env`:

```
SOONERAI_API_KEY=your_key_here
SOONERAI_BASE_URL=https://ai.sooners.us
SOONERAI_MODEL=gemma3:4b
```

---

## Run the Classifier

Run the main script:

```bash
python cifar10_classify.py
```

Artifacts created:

* **`confusion_matrix.png`** = labeled confusion matrix
* **`misclassifications.jsonl`** = raw model replies for incorrect predictions

---

## System Prompts

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
```python
SYSTEM_PROMPT = """
You are a CIFAR-10 image classifier. You must output EXACTLY ONE label from this list and nothing else:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Rules (follow strictly):
1) Output format: a single lowercase label from the list above. No punctuation, no extra words, no quotes.
2) Choose the most probable class, even if uncertain. Never invent other labels or synonyms.
3) The images are tiny (32×32), so rely on coarse cues and overall shape/color.

Output ONLY the final label.
""".strip()
```


---

## Results and Discussion

Prompt 1 = 59% accuracy
Prompt 2 = 52% accuracy

**Common Error Patterns**

* **truck ↔ automobile** – similar shape at small resolution
* **airplane ↔ bird** – small size and flying posture cause confusion
* **cat ↔ dog** – texture details are hard to distinguish in 32×32

**Ideas for Improvement**

* Use a stricter instruction enforcing the label list.
* Provide extra context (“tiny 32×32 CIFAR-10 dataset”).
* Ask the model to “think carefully but answer with only one label.”

