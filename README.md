# PuchAI-x-OpenAI
Team BITStoBYTE
# PuchAI X OpenAI Hackathon Project
## Team: **BitsToBYTE**

---


# AI-Powered Phishing Detection MCP Server

We developed an intelligent phishing detection system that combines machine learning with AI APIs through the Model Context Protocol (MCP) framework. Our solution provides real-time URL analysis to protect users from malicious websites and phishing attacks.

---

## 🛠 Tech Stack
- **Backend**: Python, FastMCP
- **ML Framework**: LightGBM, scikit-learn
- **AI Integration**: Grok AI API
- **Data Processing**: pandas, numpy
- **Authentication**: Bearer Token with RSA encryption
- **Deployment**: Async server with ngrok tunneling

---

## 📊 Dataset
**Ebbu2017 Phishing Dataset**
- **Source**: https://github.com/ebubekirbbr/pdd/tree/master/input
- **Legitimate URLs**: 36,400 samples
- **Phishing URLs**: 37,175 samples
- **Total**: 73,575 URLs for training

---

## 🔧 Step-by-Step Implementation

### Step 1: Data Preprocessing
```python
# Combined legitimate and phishing datasets
# Created binary labels (0 = legitimate, 1 = phishing)
# Saved as unified CSV format
```

**Key Actions:**
- Loaded JSON datasets for legitimate and phishing URLs
- Added binary classification labels
- Combined datasets into single training file
- Ensured data quality and consistency

### Step 2: Feature Engineering
We implemented a hybrid approach combining:

**Statistical Features (10 features):**
- URL length and hostname length
- Path length and subdirectory count
- Number of digits and special characters
- IP address detection
- HTTPS usage
- Query parameter count
- Suspicious keyword detection

**TF-IDF Features (3000 features):**
- Character-level n-grams (3-5 chars)
- Captures URL patterns and structures
- Handles obfuscation techniques

### Step 3: Model Training
**LightGBM Classifier Configuration:**
- 300 estimators for robust learning
- Learning rate: 0.05 for stability
- Unlimited depth for complex patterns
- Cross-validation with 80-20 split

**Performance Metrics:**
- High accuracy on test set
- Balanced precision and recall
- Fast inference time for real-time detection

### Step 4: MCP Server Architecture
**Core Components:**
1. **Authentication System**: Secure bearer token validation
2. **Model Loading**: Pickle-based model persistence
3. **Dual Scoring**: ML + Grok AI hybrid approach
4. **Statistics Tracking**: User and global analytics
5. **Error Handling**: Comprehensive exception management

### Step 5: API Design
**Main Tools/Endpoints:**

1. **`validate()`**: Server validation for Puch AI
2. **`analyze_url()`**: Single URL phishing analysis
3. **`batch_analyze_urls()`**: Bulk analysis (up to 10 URLs)
4. **`get_user_stats()`**: User analytics and history

### Step 6: Hybrid Scoring System
**Lambda-weighted Combination:**
- Default λ = 0.6 (60% Grok AI, 40% ML model)
- Customizable weighting per request
- Fallback mechanisms for API failures
- Confidence level classification

### Step 7: Real-time Processing
**Optimization Features:**
- Async request handling
- Processing time tracking
- Memory-efficient model loading
- Batch processing capabilities

### Step 8: User Experience
**Response Format:**
- Clear phishing/safe classification
- Risk levels (LOW/MEDIUM/HIGH)
- Confidence indicators
- Detailed technical analysis
- User-friendly summaries with emojis

### Step 9: Security & Monitoring
**Security Measures:**
- Token-based authentication
- Input validation and sanitization
- Rate limiting capabilities
- Comprehensive logging

**Analytics:**
- Per-user request tracking
- Global phishing detection rates
- Error monitoring
- Server uptime statistics

### Step 10: Deployment & Scaling
**Production Setup:**
- HTTP server on port 8086
- Ngrok tunneling for public access
- Docker-ready configuration
- Environment variable management

---

## 🎖 Key Innovations

### 1. **Hybrid AI Approach**
Combines traditional ML (LightGBM) with modern LLM capabilities (Grok AI) for superior accuracy.

### 2. **MCP Integration**
Leverages Model Context Protocol for seamless integration with AI assistants and chat interfaces.

### 3. **Real-time Analytics**
Provides instant feedback with detailed statistics and user behavior tracking.

### 4. **Scalable Architecture**
Async design supports concurrent requests with efficient resource utilization.

### 5. **Intelligent Feature Engineering**
Novel combination of statistical and TF-IDF features captures both obvious and subtle phishing indicators.

---

## 📈 Results & Impact
<img width="1127" height="560" alt="image" src="https://github.com/user-attachments/assets/4d3aa8f8-e77b-4b81-b027-1da9407de6b2" />
<img width="1151" height="625" alt="image" src="https://github.com/user-attachments/assets/eea2fb9f-13d9-4f4b-9c02-bc3c8fb512b4" />

### Model Performance
- **High Accuracy**: Robust detection across diverse phishing techniques
- **Low False Positives**: Minimizes disruption to legitimate browsing
- **Fast Response**: Sub-second analysis for real-time protection

### User Benefits
- **Instant Protection**: Real-time URL scanning
- **Batch Processing**: Efficient bulk analysis
- **Privacy-Focused**: No data storage, immediate analysis
- **User Analytics**: Track personal security awareness

### Technical Achievements
- **Production-Ready**: Comprehensive error handling and logging
- **Extensible Design**: Easy integration of new AI models
- **Secure Implementation**: Industry-standard authentication
- **Scalable Infrastructure**: Supports high-volume requests

---

## 🚀 Future Enhancements

1. **Advanced ML Models**: Integration of transformer-based models
2. **Real-time Learning**: Adaptive model updates from user feedback
3. **Mobile SDK**: Native mobile app integration
4. **Browser Extension**: Direct browser protection
5. **Enterprise Features**: Advanced analytics and admin panels

---

## 🏆 Hackathon Value Proposition

Our solution addresses a critical cybersecurity need with innovative AI integration. By combining traditional ML with modern LLM capabilities through MCP, we've created a production-ready system that can protect millions of users from phishing attacks.

**Why BitsToBYTE Deserves to Win:**
- **Technical Excellence**: Sophisticated hybrid AI approach
- **Real-world Impact**: Addresses critical cybersecurity challenges  
- **Production Quality**: Comprehensive error handling and monitoring
- **Innovation**: Novel use of MCP for security applications
- **Scalability**: Architecture ready for enterprise deployment

---
