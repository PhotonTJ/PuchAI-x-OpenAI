import React, { useState } from "react";

const styles = `
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  .analyzer-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
  }

  .analyzer-wrapper {
    width: 100%;
    max-width: 800px;
  }

  .analyzer-header {
    text-align: center;
    margin-bottom: 40px;
    color: white;
  }

  .shield-icon {
    width: 60px;
    height: 60px;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 50%;
    margin: 0 auto 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    backdrop-filter: blur(10px);
  }

  .analyzer-header h1 {
    font-size: 2.5rem;
    margin-bottom: 10px;
    font-weight: 700;
  }

  .analyzer-header p {
    font-size: 1.1rem;
    opacity: 0.9;
  }

  .analyzer-card {
    background: white;
    border-radius: 20px;
    padding: 40px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
    backdrop-filter: blur(10px);
  }

  .input-section {
    margin-bottom: 30px;
  }

  .input-label {
    display: block;
    font-weight: 600;
    margin-bottom: 10px;
    color: #374151;
    font-size: 0.9rem;
  }

  .input-container {
    display: flex;
    gap: 15px;
    align-items: stretch;
  }

  .input-wrapper {
    flex: 1;
    position: relative;
  }

  .input-icon {
    position: absolute;
    left: 15px;
    top: 50%;
    transform: translateY(-50%);
    color: #9ca3af;
    font-size: 18px;
    pointer-events: none;
  }

  .url-input {
    width: 100%;
    padding: 15px 15px 15px 45px;
    border: 2px solid #e5e7eb;
    border-radius: 12px;
    font-size: 1rem;
    transition: all 0.3s ease;
    background: #fafafa;
  }

  .url-input:focus {
    outline: none;
    border-color: #667eea;
    background: white;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
  }

  .url-input:disabled {
    background: #f3f4f6;
    color: #9ca3af;
    cursor: not-allowed;
  }

  .analyze-btn {
    padding: 15px 30px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 140px;
    justify-content: center;
  }

  .analyze-btn:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
  }

  .analyze-btn:disabled {
    background: #d1d5db;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }

  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid transparent;
    border-top: 2px solid currentColor;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .error-message {
    background: #fef2f2;
    border: 2px solid #fecaca;
    color: #dc2626;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .loading-state {
    text-align: center;
    padding: 40px;
    color: #6b7280;
  }

  .loading-content {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 15px;
  }

  .response-box {
    background: #eff6ff;
    border: 2px solid #dbeafe;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
  }

  .results-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1e40af;
    margin-bottom: 20px;
    text-align: center;
    border-bottom: 2px solid #dbeafe;
    padding-bottom: 10px;
  }

  .section {
    margin-bottom: 25px;
  }

  .section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 12px;
    border-left: 4px solid #3b82f6;
    padding-left: 12px;
    background: rgba(59, 130, 246, 0.1);
    padding: 8px 12px;
    border-radius: 6px;
  }

  .stat-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 15px;
    margin-bottom: 8px;
    background: white;
    border-radius: 8px;
    border-left: 3px solid #e5e7eb;
  }

  .stat-label {
    font-weight: 600;
    color: #374151;
    font-size: 0.95rem;
  }

  .stat-value {
    color: #1f2937;
    font-weight: 500;
    text-align: right;
    max-width: 60%;
    word-break: break-word;
  }

  .status-safe {
    color: #059669;
    font-weight: 600;
  }

  .status-danger {
    color: #dc2626;
    font-weight: 600;
  }

  .risk-low {
    background: #dcfce7;
    border-left-color: #16a34a;
    color: #166534;
    font-weight: 600;
    padding: 4px 8px;
    border-radius: 4px;
  }

  .risk-medium {
    background: #fef3c7;
    border-left-color: #d97706;
    color: #92400e;
    font-weight: 600;
    padding: 4px 8px;
    border-radius: 4px;
  }

  .risk-high {
    background: #fecaca;
    border-left-color: #dc2626;
    color: #991b1b;
    font-weight: 600;
    padding: 4px 8px;
    border-radius: 4px;
  }

  .message-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 15px;
    border-radius: 8px;
    font-family: system-ui, -apple-system, sans-serif;
    line-height: 1.6;
    white-space: pre-wrap;
    color: #334155;
    font-size: 0.9rem;
  }

  .score-display {
    font-size: 1.2rem;
    font-weight: 700;
    color: #3b82f6;
  }

  .timestamp {
    font-family: monospace;
    font-size: 0.85rem;
    color: #6b7280;
  }

  @media (max-width: 768px) {
    .input-container {
      flex-direction: column;
    }

    .analyzer-header h1 {
      font-size: 2rem;
    }

    .analyzer-card {
      padding: 25px;
    }
  }
`;

export default function MCPClient() {
  const [website, setWebsite] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!website.trim()) {
      setError("Please enter a valid website URL");
      return;
    }

    // Basic URL validation
    try {
      new URL(website.startsWith('http') ? website : `https://${website}`);
    } catch {
      setError("Please enter a valid URL format");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const urlToAnalyze = website.startsWith('http') ? website : `https://${website}`;
      const res = await fetch("https://adfd3dbf5643.ngrok-free.app/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: urlToAnalyze, puch_user_id: "U001" })
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to analyze website");
    } finally {
      setLoading(false);
    }
  };

  const getRiskData = (score) => {
    if (score < 0.3) {
      return {
        level: "Low Risk",
        class: "low",
        icon: "✅"
      };
    }
    if (score < 0.7) {
      return {
        level: "Medium Risk", 
        class: "medium",
        icon: "⚠️"
      };
    }
    return {
      level: "High Risk",
      class: "high",
      icon: "❌"
    };
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !loading) {
      handleAnalyze();
    }
  };

  

    return (
    <>
      <style dangerouslySetInnerHTML={{ __html: styles }} />
      <div className="analyzer-container">
        <div className="analyzer-wrapper">
          <div className="analyzer-header">
            <div className="shield-icon">🛡️</div>
            <h1>Website Risk Analyzer</h1>
            <p>Analyze websites for potential security risks and threats</p>
          </div>

          <div className="analyzer-card">
            <div className="input-section">
              <label htmlFor="website-url" className="input-label">
                Website URL
              </label>
              <div className="input-container">
                <div className="input-wrapper">
                  <span className="input-icon">🌐</span>
                  <input
                    id="website-url"
                    type="text"
                    className="url-input"
                    placeholder="https://example.com or example.com"
                    value={website}
                    onChange={(e) => setWebsite(e.target.value)}
                    disabled={loading}
                  />
                </div>
                <button
                  onClick={handleAnalyze}
                  disabled={loading || !website.trim()}
                  className="analyze-btn"
                >
                  {loading ? (
                    <>
                      <div className="spinner"></div>
                      Analyzing...
                    </>
                  ) : (
                    "Analyze"
                  )}
                </button>
              </div>
            </div>

            {error && (
              <div className="error-message">
                <span>❌</span>
                <span>{error}</span>
              </div>
            )}

            {loading && (
              <div className="loading-state">
                <div className="loading-content">
                  <div className="spinner"></div>
                  <span>Analyzing website security...</span>
                </div>
              </div>
            )}

            {result && (
              <div className="response-box">
                <h2 className="results-header">🔍 Security Analysis Report</h2>
                
                <div className="section">
                  <h3 className="section-title">📋 Overall Status</h3>
                  <div className="stat-item">
                    <span className="stat-label">Analysis Status:</span>
                    <span className={`stat-value ${result.success ? 'status-safe' : 'status-danger'}`}>
                      {result.success ? '✅ Complete' : '❌ Failed'}
                    </span>
                  </div>
                </div>

                <div className="section">
                  <h3 className="section-title">📝 Analysis Summary</h3>
                  <div className="message-box">
                    {result.message}
                  </div>
                </div>
                
                {result.data && (
                  <>
                    <div className="section">
                      <h3 className="section-title">🌐 Website Information</h3>
                      <div className="stat-item">
                        <span className="stat-label">Analyzed URL:</span>
                        <span className="stat-value" style={{ fontFamily: 'monospace' }}>{result.data.url}</span>
                      </div>
                    </div>

                    <div className="section">
                      <h3 className="section-title">🛡️ Security Assessment</h3>
                      <div className="stat-item">
                        <span className="stat-label">Phishing Detection:</span>
                        <span className={`stat-value ${result.data.is_phishing ? 'status-danger' : 'status-safe'}`}>
                          {result.data.is_phishing ? '⚠️ Phishing Detected' : '✅ No Phishing'}
                        </span>
                      </div>
                      
                      <div className="stat-item">
                        <span className="stat-label">Risk Level:</span>
                        <span className={`stat-value risk-${result.data.risk_level.toLowerCase()}`}>
                          {result.data.risk_level === 'LOW' && '🟢 '} 
                          {result.data.risk_level === 'MEDIUM' && '🟡 '} 
                          {result.data.risk_level === 'HIGH' && '🔴 '} 
                          {result.data.risk_level}
                        </span>
                      </div>
                      
                      <div className="stat-item">
                        <span className="stat-label">Confidence Level:</span>
                        <span className="stat-value" style={{ textTransform: 'capitalize' }}>
                          {result.data.confidence}
                        </span>
                      </div>
                      
                      <div className="stat-item">
                        <span className="stat-label">Security Score:</span>
                        <span className="stat-value score-display">
                          {result.data.final_score}/1.0
                        </span>
                      </div>
                    </div>
                    
                    {result.data.metadata && (
                      <div className="section">
                        <h3 className="section-title">⚙️ Technical Details</h3>
                        <div className="stat-item">
                          <span className="stat-label">Processing Time:</span>
                          <span className="stat-value">{result.data.metadata.processing_time_ms}ms</span>
                        </div>
                        
                        <div className="stat-item">
                          <span className="stat-label">Analysis Timestamp:</span>
                          <span className="stat-value timestamp">
                            {new Date(result.data.metadata.timestamp).toLocaleString()}
                          </span>
                        </div>

                        
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}