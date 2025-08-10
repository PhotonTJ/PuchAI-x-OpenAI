import React, { useState } from "react";
import "./MCPClient.css";

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

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch("https://0d21883478d2.ngrok-free.app/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: website, puch_user_id: "U001" })
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (score) => {
    if (score < 0.3) return "green";
    if (score < 0.7) return "orange";
    return "red";
  };

  return (
    <div className="container">
      <div className="card">
        <h1>Website Risk Analyzer</h1>
        <div className="input-row">
          <input
            type="text"
            placeholder="https://example.com"
            value={website}
            onChange={(e) => setWebsite(e.target.value)}
          />
          <button onClick={handleAnalyze} disabled={loading}>
            {loading ? "Analyzing..." : "Submit"}
          </button>
        </div>

        {error && <p className="error">❌ {error}</p>}

        {result && (
          <div className="result">
            <p
              className="score"
              style={{ color: getRiskColor(result.score) }}
            >
              {result.score.toFixed(2)}/1.0
            </p>
            <p>
              <strong>Risk Level:</strong>{" "}
              {result.score < 0.3
                ? "Low"
                : result.score < 0.7
                ? "Medium"
                : "High"}
            </p>
            <p>
              <strong>Confidence:</strong> {result.confidence}%
            </p>
            <p className="description">{result.description}</p>
          </div>
        )}
      </div>
    </div>
  );
}
