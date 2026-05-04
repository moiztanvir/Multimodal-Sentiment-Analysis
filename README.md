# 🚀 Multimodal Sentiment Analysis using HCGAF & OCR Fusion

> Enhancing sentiment understanding by bridging the gap between **text, images, and embedded visual text (OCR)** using a **Hierarchical Cross-Guided Attention Fusion (HCGAF)** framework.

---

## 📌 Overview

Traditional sentiment analysis relies only on text, but modern social media (memes, posts, ads) combines **images + captions + embedded text**.

This project introduces a **multimodal sentiment analysis system** that:

* Extracts features from **text and images**
* Uses **OCR to capture hidden textual context inside images**
* Applies a **Hierarchical Cross-Guided Attention Fusion (HCGAF)** architecture
* Improves classification performance significantly over baseline models

---

## 🧠 Key Features

* 🔍 **OCR Integration (PyTesseract)**
  Extracts hidden text from images (e.g., memes)

* 🧩 **Multimodal Fusion (HCGAF)**
  Cross-attention mechanism between text and image features

* 🏗 **Hierarchical Classification**

  * Stage A: Negative vs Non-Negative
  * Stage B: Positive vs Neutral

* 🤖 **Transformer-based Models**

  * Text: DeBERTa-v3
  * Vision: CLIP (ViT-B/32)

* 🛡 **Advanced Regularization**

  * R-Drop
  * Modality Dropout

* 📊 **Explainability (XAI)**

  * SHAP
  * LIME

---

## 📂 Dataset

* **MVSA-Single Dataset**

  * Image + Caption pairs
  * Labels: `Positive`, `Neutral`, `Negative`

⚠️ Challenge:

* **Label Noise** (text and image sentiment may conflict)

---

## ⚙️ Architecture

### 🔄 Pipeline Workflow

```
          +-------------------+
          |  Input Data       |
          | (Image + Caption) |
          +--------+----------+
                   |
        +----------v----------+
        |   OCR Extraction    |
        +----------+----------+
                   |
     +-------------v-------------+
     | Text Encoder (DeBERTa-v3) |
     +-------------+-------------+
                   |
     +-------------v-------------+
     | Vision Encoder (CLIP ViT) |
     +-------------+-------------+
                   |
        +----------v----------+
        | Cross-Guided        |
        | Attention Fusion    |
        +----------+----------+
                   |
        +----------v----------+
        | Hierarchical        |
        | Classifier          |
        +----------+----------+
                   |
              Sentiment
```

---

## 🧪 Baseline vs Proposed Model

| Model                    | Accuracy   | Macro-F1   |
| ------------------------ | ---------- | ---------- |
| BERT + ResNet (Baseline) | 68.40%     | 64.20%     |
| HCGAF + OCR (Proposed)   | **81.24%** | **79.51%** |

✔️ **~11% improvement in accuracy**
✔️ Significant boost in **negative sentiment detection**

---

## 🔬 Ablation Study

| Variant         | Accuracy | Macro-F1 |
| --------------- | -------- | -------- |
| Full Model      | 81.2%    | 79.5%    |
| w/o OCR         | 76.4%    | 74.1%    |
| w/o R-Drop      | 78.1%    | 76.2%    |
| w/o SupCon Loss | 77.5%    | 75.8%    |

📌 **Insight:**
OCR contributed the **largest performance gain (~5%)**

---

## 🧠 Why This Works

* 👁 **CLIP aligns image-text representations**
* 🧾 **OCR captures hidden sentiment in images**
* 🔁 **Cross-attention enables modality interaction**
* 🧩 **Hierarchical classification simplifies learning**
* ⚖️ **Focal Loss improves class imbalance handling**

---

## 🛠 Tech Stack

* **Frameworks:** PyTorch, HuggingFace Transformers
* **Vision Models:** CLIP (ViT-B/32)
* **NLP Models:** DeBERTa-v3
* **OCR:** PyTesseract
* **Explainability:** SHAP, LIME
* **Environment:** Google Colab / Linux (GPU: T4 / A100)

---

## 🚀 Installation

```bash
git clone https://github.com/moiztanvir/Multimodal-Sentiment-Analysis.git
cd Multimodal-Sentiment-Analysis

pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Train Model

```bash
python train.py
```

### 2. Evaluate

```bash
python evaluate.py
```

### 3. Run Inference

```bash
python predict.py --image path/to/image.jpg --text "Sample caption"
```

---

## 📊 Results & Visualizations

* 📈 Training Curves
* 🔲 Confusion Matrices (Stage A & B)
* 🔍 SHAP Feature Importance
* 🧠 LIME Explanations

---

## 🔍 Explainability Insights

* **LIME:** Model focuses on sarcastic OCR words
* **SHAP:**

  * Image features → strong for *Negative*
  * Text features → dominant for *Neutral*

---

## ⚠️ Limitations

* 💻 High computational cost (CLIP + DeBERTa)
* 🧾 OCR noise in low-quality images
* 📦 Large model size (not mobile-friendly)

---

## 🔮 Future Work

* ⚡ Model compression (Distillation / Quantization)
* 📱 Mobile deployment
* 🧠 Better OCR filtering
* 🔊 Extend to audio (multimodal tri-fusion)

---

## 📖 References

* MVSA Dataset
* CLIP (OpenAI)
* DeBERTa (ICLR)
* R-Drop (NeurIPS)
* Supervised Contrastive Learning
* SHAP & LIME (XAI)

---

## 👨‍💻 Authors

* **Moiz Tanvir**
* Ahtezaz Ahsan
* Amna Shahid

---

## ⭐ Contributing

Contributions are welcome!
Feel free to open issues or submit pull requests.

---

## 📜 License

This project is for academic/research purposes.

---

