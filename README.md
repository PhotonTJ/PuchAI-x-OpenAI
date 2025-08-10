# 🛡️ Smart Phishing Protection System
**Team BitsToBYTE** - *PuchAI x OpenAI Hackathon*

---

## 🎯 The Problem We Solved

**Imagine this scenario:** You receive an email claiming to be from your bank asking you to "verify your account." You click the link, and it looks legitimate, but it's actually a fake website designed to steal your password and personal information.

**This is called phishing**, and it happens to millions of people every day. In 2024 alone, phishing attacks increased by 58%, costing individuals and businesses billions of dollars.

**Our solution?** We built an AI-powered guardian that can instantly tell if a website is trying to trick you.

---

## 🤖 What We Built: Your Digital Bodyguard

Think of our system as having a super-smart friend who:
- **Analyzes websites in seconds** - Just give it a web address (URL)
- **Spots danger signs humans miss** - Detects 15+ suspicious patterns
- **Never gets tired** - Works 24/7 protecting you
- **Learns from experience** - Gets smarter with every website it checks

### How It Works (In Simple Terms):

1. **📝 You share a suspicious link** with our system
2. **🔍 Our AI examines everything** about that website instantly:
   - Is the web address spelled strangely?
   - Does it try to copy famous brands?
   - Are there hidden tricks in the code?
3. **⚡ In under 1 second**, you get a clear answer:
   - ✅ **SAFE** - Go ahead and visit
   - ⚠️ **SUSPICIOUS** - Be careful
   - 🚫 **DANGEROUS** - Don't visit this site!

---

## 🧠 The "Secret Sauce": Two AI Brains Working Together

Most security tools use just one method to detect threats. We use **two different AI systems** that work as a team:

### 🤖 **Brain #1: The Pattern Detective**
- Trained on 73,000+ websites (both safe and dangerous)
- Spots mathematical patterns that indicate phishing
- Like a detective who notices tiny details others miss

### 🤖 **Brain #2: The Smart Assistant** 
- Uses advanced language AI (like ChatGPT's cousin)
- Understands context and meaning
- Like having a cybersecurity expert analyze each site

**Together, they're more accurate than either could be alone!**

---

## 📊 Real Impact & Results

### What Our System Protects You From:
- **Fake banking websites** that steal your login credentials
- **Counterfeit shopping sites** that take your credit card info
- **Malicious downloads** disguised as legitimate software
- **Social media scams** that hijack your accounts

### Performance That Matters:
- ⚡ **Lightning Fast**: Results in under 1 second
- 🎯 **Highly Accurate**: Catches threats while avoiding false alarms
- 📱 **Always Available**: Works on phones, computers, and tablets
- 🔒 **Privacy-First**: We don't store or track your browsing

---

## 🌟 Why This Matters for Everyone

### For Individuals:
- **Save Money**: Avoid financial scams and identity theft
- **Peace of Mind**: Browse confidently knowing you're protected
- **Learn & Improve**: Understand what makes websites dangerous

### For Businesses:
- **Protect Employees**: Prevent costly security breaches
- **Batch Analysis**: Check multiple links at once
- **Integration Ready**: Works with existing security systems

### For Society:
- **Reduce Cybercrime**: Make phishing less profitable for criminals
- **Digital Literacy**: Help people recognize threats
- **Accessible Security**: Advanced protection for everyone, not just tech experts

---

## 🚀 The Innovation That Sets Us Apart

### 1. **Hybrid Intelligence**
We don't just use one AI model - we combine multiple approaches for maximum accuracy, like having both a microscope and a telescope to see threats clearly.

### 2. **Real-Time Protection**
While other tools might take minutes to analyze a website, ours works instantly - because in cybersecurity, every second counts.

### 3. **Human-Friendly Results**
Instead of technical jargon, you get clear, actionable advice with easy-to-understand risk levels and emoji indicators.

### 4. **Batch Processing Power**
Need to check 10 suspicious emails at once? No problem - our system handles bulk analysis efficiently.

---



## 🏆 Why Team BitsToBYTE Should Win

### 💡 **Innovation**: 
We combined cutting-edge AI technologies in a novel way that's both powerful and practical.

### 🌍 **Real-World Impact**: 
Our solution addresses a problem that affects millions of people daily, potentially saving them money, privacy, and stress.

### 🛠️ **Technical Excellence**: 
Built a production-ready system that's both sophisticated under the hood and simple to use.

### 🚀 **Scalability**: 
Our architecture can protect individuals today and scale to protect entire organizations tomorrow.

### ❤️ **Human-Centered Design**: 
Created technology that serves people, not the other way around - making advanced cybersecurity accessible to everyone.

---

## 💬 The Bottom Line

**Phishing attacks are getting smarter, but so is our defense.**

We built an AI system that thinks like a cybersecurity expert but explains things like a helpful friend. It's fast, accurate, and designed for real people facing real threats online.

In a world where digital safety shouldn't be a luxury, we're making advanced protection available to everyone, one URL at a time.

# Phishing URL Detection with LightGBM

## Overview

This repository offers a robust framework for detecting phishing URLs using advanced machine learning techniques. The solution combines LightGBM for high-performance classification, TF-IDF for effective feature extraction from URLs, and a suite of preprocessing steps to maximize detection accuracy. The project is designed for researchers, data scientists, and security professionals seeking to experiment, train, and deploy phishing detection models with ease.

## Repository Structure

- **Data Pre-Processing/**  
    Contains scripts and utilities for cleaning and transforming raw datasets. Includes feature engineering modules, normalization routines, and functions for handling missing or anomalous data. This folder ensures that input data is properly formatted and enriched before model training.

- **Datasets-Phishing/**  
    Stores curated datasets of phishing and legitimate URLs. Includes raw data files, processed datasets ready for training, and test sets for validation. Documentation within this folder describes dataset sources and formats.

- **MCP/**  
    Houses trained LightGBM models, TF-IDF vectorizers, and core detection scripts such as `phishing_mcp_detector.py`. This folder is intended for inference and deployment, providing all necessary artifacts to run URL detection on new samples.

- **Training/**  
    Contains Jupyter notebooks for exploratory data analysis, model training scripts, hyperparameter tuning experiments, and saved model checkpoints. This folder supports reproducibility and further research, allowing users to retrain or improve models.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/phishing-url-detection.git
    cd phishing-url-detection
    ```

2. **Set up Python environment (recommended: Python 3.8+):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    **Main requirements:**
    - lightgbm: Gradient boosting framework for fast, accurate classification.
    - scikit-learn: Essential for preprocessing, model evaluation, and TF-IDF vectorization.
    - pandas: Data manipulation and analysis.
    - numpy: Efficient numerical computations.

## Usage

### 1. Preprocess the Dataset

Prepare your raw URL dataset for training by running the preprocessing script. This step cleans the data, extracts relevant features, and outputs a processed CSV file.

```bash
python Data\ Pre-Processing/preprocess.py --input Datasets-Phishing/raw.csv --output Datasets-Phishing/processed.csv
```
- `--input`: Path to the raw dataset.
- `--output`: Path to save the processed dataset.

### 2. Train the Model

Train the LightGBM model using the processed dataset. The script will save the trained model and TF-IDF vectorizer for later use.

```bash
python Training/train_model.py --data Datasets-Phishing/processed.csv --output MCP/model.pkl
```
- `--data`: Path to the processed dataset.
- `--output`: Path to save the trained model.

### 3. Run URL Detection

Detect whether a given URL is phishing or legitimate using the trained model and vectorizer.

```bash
python MCP/phishing_mcp_detector.py --model MCP/model.pkl --vectorizer MCP/tfidf_vectorizer.pkl --url "http://example.com"
```
- `--model`: Path to the trained LightGBM model.
- `--vectorizer`: Path to the TF-IDF vectorizer.
- `--url`: URL to be classified.

## Dependencies

This project relies on the following Python packages:
- **lightgbm**: For training and inference of the phishing detection model.
- **scikit-learn**: Used for TF-IDF vectorization, preprocessing, and evaluation metrics.
- **pandas**: For data loading, manipulation, and analysis.
- **numpy**: For efficient numerical operations and array handling.

All dependencies are listed in `requirements.txt` for easy installation.

## Example

Below is a Python snippet demonstrating how to use the detector programmatically:

```python
from MCP.phishing_mcp_detector import detect_url

result = detect_url(
    url="http://suspicious-url.com",
    model_path="MCP/model.pkl",
    vectorizer_path="MCP/tfidf_vectorizer.pkl"
)
print("Phishing" if result else "Legitimate")
```
- Replace the URL and paths as needed.
- The function returns `True` for phishing, `False` for legitimate URLs.

## Contributing

We welcome contributions from the community! You can help by:
- Reporting bugs or issues.
- Suggesting new features or improvements.
- Submitting pull requests with code enhancements or documentation updates.

Please review [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on how to get involved, coding standards, and the review process.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this software in accordance with the license terms.
## Technical Details

### LightGBM for Phishing Detection

LightGBM is a gradient boosting framework that builds decision trees in a highly efficient, parallelizable manner. In this project, LightGBM is trained on engineered features extracted from URLs to classify them as phishing or legitimate. Its ability to handle large datasets and categorical features makes it well-suited for this task, providing fast training and high accuracy.

### TF-IDF Vectorization of URL Features

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert URLs into numerical feature vectors. Each URL is tokenized (split into meaningful parts such as domain, path, and query parameters), and TF-IDF assigns weights to these tokens based on their frequency and uniqueness across the dataset. This helps the model capture patterns commonly found in phishing URLs, such as suspicious keywords or unusual structures.

### Feature Engineering in Preprocessing

The preprocessing scripts and notebooks perform several feature engineering steps, including:
- **Tokenization:** Breaking URLs into components (subdomains, paths, file extensions).
- **Length-based features:** Calculating URL length, number of dots, and special characters.
- **Keyword extraction:** Identifying suspicious terms (e.g., "login", "secure", "update").
- **Entropy calculation:** Measuring randomness in URL strings to detect obfuscation.
- **Presence of IP addresses:** Flagging URLs that use IP addresses instead of domain names.

These features are combined with TF-IDF vectors to create a comprehensive input for the model.

### Model Evaluation Metrics

Model performance is assessed using standard classification metrics:
- **Accuracy:** Proportion of correctly classified URLs.
- **Precision:** Fraction of predicted phishing URLs that are actually phishing.
- **Recall:** Fraction of actual phishing URLs correctly identified.
- **F1 Score:** Harmonic mean of precision and recall, balancing false positives and false negatives.
- **ROC-AUC:** Measures the model's ability to distinguish between classes across different thresholds.

These metrics are reported in the training notebooks to guide model selection and tuning.
## Architecture and Workflow

The following diagram illustrates the end-to-end workflow for phishing URL detection:

```
Dataset
    │
    ▼
Preprocessing
    │
    ▼
Feature Extraction (TF-IDF)
    │
    ▼
Model Training (LightGBM)
    │
    ▼
Model Storage
    │
    ▼
Detection Script (phishing_mcp_detector.py)
    │
    ▼
Output (Phishing / Legitimate)
```

### Stage Explanations

- **Dataset:** Raw URLs (phishing and legitimate) are collected and stored for analysis.
- **Preprocessing:** Data cleaning, normalization, and feature engineering are performed to prepare the URLs for modeling.
- **Feature Extraction (TF-IDF):** URLs are tokenized and transformed into numerical vectors using TF-IDF, capturing important textual patterns.
- **Model Training (LightGBM):** The processed features are used to train a LightGBM classifier to distinguish phishing from legitimate URLs.
- **Model Storage:** The trained model and TF-IDF vectorizer are saved for future inference.
- **Detection Script:** `phishing_mcp_detector.py` loads the model and vectorizer to classify new URLs.
- **Output:** The script outputs whether the input URL is phishing or legitimate.


**Team BitsToBYTE: Protecting your digital life with AI that cares.**
