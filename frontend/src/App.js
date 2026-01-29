import React, { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement,
  Filler,
} from 'chart.js';
import { Bar, Doughnut, Line } from 'react-chartjs-2';
import { FaBrain, FaUpload, FaChartLine, FaShieldAlt, FaHistory, FaMicroscope, FaHeartbeat, FaCheckCircle, FaTimesCircle, FaInfoCircle } from 'react-icons/fa';
import './App.css';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement,
  Filler
);

const API_URL = 'http://localhost:5000';

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [animatedConfidence, setAnimatedConfidence] = useState(0);
  const [systemStatus, setSystemStatus] = useState('checking');

  // Check system health on mount
  useEffect(() => {
    checkSystemHealth();
  }, []);

  const checkSystemHealth = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/health`);
      setSystemStatus(response.data.status === 'healthy' ? 'online' : 'offline');
    } catch (err) {
      setSystemStatus('offline');
    }
  };

  // Animate confidence when result changes
  useEffect(() => {
    if (result && result.success) {
      const targetConfidence = result.confidence;
      let current = 0;
      const increment = targetConfidence / 50;
      const timer = setInterval(() => {
        current += increment;
        if (current >= targetConfidence) {
          setAnimatedConfidence(targetConfidence);
          clearInterval(timer);
        } else {
          setAnimatedConfidence(current);
        }
      }, 20);
      return () => clearInterval(timer);
    }
  }, [result]);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setImage(file);
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(file);
      setResult(null);
      setError(null);
      setAnimatedConfidence(0);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png'] },
    multiple: false,
  });

  const analyzeImage = async () => {
    if (!image) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', image);

    try {
      const response = await axios.post(`${API_URL}/api/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const newResult = response.data;
      setResult(newResult);

      if (newResult.success) {
        // Add to history
        const historyItem = {
          id: Date.now(),
          timestamp: new Date().toLocaleString(),
          prediction: newResult.predicted_class,
          confidence: newResult.confidence,
          preview: preview,
          probabilities: newResult.all_probabilities,
        };
        setHistory((prev) => [historyItem, ...prev.slice(0, 9)]);
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Analysis failed. Please ensure the server is running.');
      setResult({ success: false, error: err.response?.data?.error || 'Analysis failed' });
    } finally {
      setLoading(false);
    }
  };

  const resetAnalysis = () => {
    setImage(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setAnimatedConfidence(0);
  };

  // Chart configurations
  const getBarChartData = () => {
    if (!result?.all_probabilities) return null;
    
    const labels = Object.keys(result.all_probabilities);
    const data = Object.values(result.all_probabilities);
    
    return {
      labels: labels.map(l => formatClassName(l)),
      datasets: [
        {
          label: 'Probability (%)',
          data: data,
          backgroundColor: [
            'rgba(16, 185, 129, 0.85)',
            'rgba(59, 130, 246, 0.85)',
            'rgba(245, 158, 11, 0.85)',
            'rgba(239, 68, 68, 0.85)',
          ],
          borderColor: [
            'rgb(16, 185, 129)',
            'rgb(59, 130, 246)',
            'rgb(245, 158, 11)',
            'rgb(239, 68, 68)',
          ],
          borderWidth: 2,
          borderRadius: 8,
        },
      ],
    };
  };

  const getDoughnutData = () => {
    if (!result?.all_probabilities) return null;
    
    return {
      labels: Object.keys(result.all_probabilities).map(l => formatClassName(l)),
      datasets: [
        {
          data: Object.values(result.all_probabilities),
          backgroundColor: [
            'rgba(16, 185, 129, 0.9)',
            'rgba(59, 130, 246, 0.9)',
            'rgba(245, 158, 11, 0.9)',
            'rgba(239, 68, 68, 0.9)',
          ],
          borderColor: '#ffffff',
          borderWidth: 3,
          hoverOffset: 10,
        },
      ],
    };
  };

  const getLineChartData = () => {
    if (history.length === 0) return null;
    
    const recentHistory = history.slice(0, 7).reverse();
    
    return {
      labels: recentHistory.map((_, i) => `Scan ${i + 1}`),
      datasets: [
        {
          label: 'Confidence Trend (%)',
          data: recentHistory.map(h => h.confidence),
          fill: true,
          borderColor: 'rgb(16, 185, 129)',
          backgroundColor: 'rgba(16, 185, 129, 0.15)',
          tension: 0.4,
          pointBackgroundColor: 'rgb(16, 185, 129)',
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          pointRadius: 6,
          pointHoverRadius: 8,
        },
      ],
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleFont: { size: 14, weight: 'bold' },
        bodyFont: { size: 13 },
        padding: 12,
        cornerRadius: 8,
        callbacks: {
          label: (context) => `${context.parsed.y.toFixed(1)}%`,
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        grid: { color: 'rgba(0, 0, 0, 0.06)' },
        ticks: { 
          color: '#64748b',
          font: { size: 11 },
          callback: (value) => value + '%',
        },
      },
      x: {
        grid: { display: false },
        ticks: { 
          color: '#64748b',
          font: { size: 11, weight: '500' },
        },
      },
    },
  };

  const lineChartOptions = {
    ...chartOptions,
    plugins: {
      ...chartOptions.plugins,
      legend: {
        display: true,
        position: 'top',
        labels: {
          padding: 20,
          font: { size: 12, weight: '500' },
          color: '#374151',
          usePointStyle: true,
        },
      },
    },
  };

  const doughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          padding: 20,
          font: { size: 12, weight: '500' },
          color: '#374151',
          usePointStyle: true,
        },
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleFont: { size: 14, weight: 'bold' },
        bodyFont: { size: 13 },
        padding: 12,
        cornerRadius: 8,
        callbacks: {
          label: (context) => `${context.label}: ${context.parsed.toFixed(1)}%`,
        },
      },
    },
    cutout: '60%',
  };

  const formatClassName = (name) => {
    if (name === 'notumor') return 'No Tumor';
    return name.charAt(0).toUpperCase() + name.slice(1);
  };

  const getTumorInfo = (prediction) => {
    const info = {
      glioma: {
        severity: 'High',
        color: '#EF4444',
        bgColor: 'rgba(239, 68, 68, 0.1)',
        description: 'Gliomas are tumors that arise from glial cells in the brain or spine. They are the most common type of primary brain tumor.',
        recommendation: 'Immediate consultation with a neuro-oncologist is strongly recommended for further evaluation and treatment planning.',
        icon: '⚠️',
      },
      meningioma: {
        severity: 'Moderate',
        color: '#F59E0B',
        bgColor: 'rgba(245, 158, 11, 0.1)',
        description: 'Meningiomas are tumors that form on membranes covering the brain and spinal cord. Most are slow-growing and benign.',
        recommendation: 'Schedule a follow-up MRI and consult with a neurosurgeon for monitoring and treatment options.',
        icon: '⚡',
      },
      pituitary: {
        severity: 'Moderate',
        color: '#3B82F6',
        bgColor: 'rgba(59, 130, 246, 0.1)',
        description: 'Pituitary tumors are abnormal growths in the pituitary gland. Most are non-cancerous (benign) adenomas.',
        recommendation: 'Endocrinological evaluation recommended along with regular monitoring through MRI scans.',
        icon: '📋',
      },
      notumor: {
        severity: 'None',
        color: '#10B981',
        bgColor: 'rgba(16, 185, 129, 0.1)',
        description: 'No tumor detected in the MRI scan. The brain tissue appears normal in this analysis.',
        recommendation: 'Continue regular health check-ups and maintain a healthy lifestyle.',
        icon: '✅',
      },
    };
    return info[prediction?.toLowerCase()] || info.notumor;
  };

  const getConfidenceLevel = (conf) => {
    if (conf >= 90) return { text: 'Very High', color: '#10b981' };
    if (conf >= 75) return { text: 'High', color: '#3b82f6' };
    if (conf >= 60) return { text: 'Moderate', color: '#f59e0b' };
    return { text: 'Low', color: '#ef4444' };
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <div className="logo-icon">
              <FaBrain />
            </div>
            <div className="logo-text">
              <h1>NeuroScan AI</h1>
              <span>Advanced Brain Tumor Detection</span>
            </div>
          </div>
          <nav className="nav-links">
            <button 
              className={`nav-btn ${!showHistory ? 'active' : ''}`}
              onClick={() => setShowHistory(false)}
            >
              <FaMicroscope /> Analysis
            </button>
            <button 
              className={`nav-btn ${showHistory ? 'active' : ''}`}
              onClick={() => setShowHistory(true)}
            >
              <FaHistory /> History ({history.length})
            </button>
            <div className={`status-indicator ${systemStatus}`}>
              <span className="status-dot"></span>
              {systemStatus === 'online' ? 'System Online' : systemStatus === 'checking' ? 'Connecting...' : 'Offline'}
            </div>
          </nav>
        </div>
      </header>

      <main className="main-content">
        {!showHistory ? (
          <>
            {/* Stats Banner */}
            <div className="stats-banner">
              <div className="stat-card">
                <div className="stat-icon-wrapper green">
                  <FaShieldAlt className="stat-icon" />
                </div>
                <div className="stat-info">
                  <span className="stat-value">96.95%</span>
                  <span className="stat-label">Model Accuracy</span>
                </div>
              </div>
              <div className="stat-card">
                <div className="stat-icon-wrapper blue">
                  <FaChartLine className="stat-icon" />
                </div>
                <div className="stat-info">
                  <span className="stat-value">4</span>
                  <span className="stat-label">Detection Classes</span>
                </div>
              </div>
              <div className="stat-card">
                <div className="stat-icon-wrapper purple">
                  <FaHeartbeat className="stat-icon" />
                </div>
                <div className="stat-info">
                  <span className="stat-value">{history.length}</span>
                  <span className="stat-label">Scans Analyzed</span>
                </div>
              </div>
              <div className="stat-card">
                <div className="stat-icon-wrapper orange">
                  <FaBrain className="stat-icon" />
                </div>
                <div className="stat-info">
                  <span className="stat-value">MobileNetV2</span>
                  <span className="stat-label">AI Architecture</span>
                </div>
              </div>
            </div>

            {/* Main Analysis Section */}
            <div className="analysis-section">
              {/* Upload Panel */}
              <div className="panel upload-panel">
                <div className="panel-header">
                  <h2><FaUpload /> Upload MRI Scan</h2>
                  <span className="panel-subtitle">Drag & drop or click to upload</span>
                </div>
                <div className="panel-content">
                  <div
                    {...getRootProps()}
                    className={`dropzone ${isDragActive ? 'active' : ''} ${preview ? 'has-image' : ''}`}
                  >
                    <input {...getInputProps()} />
                    {preview ? (
                      <div className="preview-container">
                        <img src={preview} alt="MRI Preview" className="preview-image" />
                        {loading && (
                          <div className="scanning-overlay">
                            <div className="scan-line"></div>
                            <span>Analyzing...</span>
                          </div>
                        )}
                        <div className="preview-overlay">
                          <span>Click or drag to replace</span>
                        </div>
                      </div>
                    ) : (
                      <div className="dropzone-content">
                        <div className="dropzone-icon">
                          <FaBrain />
                        </div>
                        <p className="dropzone-text">
                          {isDragActive ? 'Drop the MRI scan here' : 'Drag & drop an MRI scan'}
                        </p>
                        <span className="dropzone-hint">or click to browse files</span>
                        <span className="dropzone-formats">Supports: JPG, JPEG, PNG</span>
                      </div>
                    )}
                  </div>

                  {image && (
                    <div className="file-info">
                      <span className="file-name">{image.name}</span>
                      <span className="file-size">{(image.size / 1024).toFixed(1)} KB</span>
                    </div>
                  )}

                  <div className="action-buttons">
                    <button
                      className="btn btn-primary"
                      onClick={analyzeImage}
                      disabled={!image || loading}
                    >
                      {loading ? (
                        <>
                          <span className="spinner"></span>
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <FaMicroscope /> Analyze Scan
                        </>
                      )}
                    </button>
                    {(image || result) && (
                      <button className="btn btn-secondary" onClick={resetAnalysis}>
                        Reset
                      </button>
                    )}
                  </div>

                  {error && (
                    <div className="error-message">
                      <FaTimesCircle /> {error}
                    </div>
                  )}
                </div>
              </div>

              {/* Results Panel */}
              <div className="panel results-panel">
                <div className="panel-header">
                  <h2><FaChartLine /> Analysis Results</h2>
                  <span className="panel-subtitle">AI-powered diagnosis</span>
                </div>
                <div className="panel-content">
                  {result && result.success ? (
                    <div className="results-content">
                      {/* Diagnosis Card */}
                      <div 
                        className="diagnosis-card"
                        style={{ 
                          borderLeftColor: getTumorInfo(result.predicted_class).color,
                          backgroundColor: getTumorInfo(result.predicted_class).bgColor,
                        }}
                      >
                        <div className="diagnosis-header">
                          <span className="diagnosis-icon">{getTumorInfo(result.predicted_class).icon}</span>
                          <span 
                            className="severity-badge"
                            style={{ backgroundColor: getTumorInfo(result.predicted_class).color }}
                          >
                            {getTumorInfo(result.predicted_class).severity} Risk
                          </span>
                        </div>
                        <h3 className="diagnosis-title" style={{ color: getTumorInfo(result.predicted_class).color }}>
                          {formatClassName(result.predicted_class)}
                        </h3>
                        <p className="diagnosis-description">
                          {getTumorInfo(result.predicted_class).description}
                        </p>
                      </div>

                      {/* Confidence Meter */}
                      <div className="confidence-section">
                        <div className="confidence-header">
                          <span>Confidence Level</span>
                          <div className="confidence-stats">
                            <span 
                              className="confidence-level-badge"
                              style={{ color: getConfidenceLevel(result.confidence).color }}
                            >
                              {getConfidenceLevel(result.confidence).text}
                            </span>
                            <span className="confidence-value">{animatedConfidence.toFixed(1)}%</span>
                          </div>
                        </div>
                        <div className="confidence-bar">
                          <div 
                            className="confidence-fill"
                            style={{ 
                              width: `${animatedConfidence}%`,
                              backgroundColor: getTumorInfo(result.predicted_class).color
                            }}
                          >
                            <div className="confidence-glow"></div>
                          </div>
                        </div>
                      </div>

                      {/* Recommendation */}
                      <div className="recommendation-card">
                        <FaCheckCircle className="rec-icon" style={{ color: getTumorInfo(result.predicted_class).color }} />
                        <div>
                          <strong>Clinical Recommendation</strong>
                          <p>{getTumorInfo(result.predicted_class).recommendation}</p>
                        </div>
                      </div>

                      {/* Disclaimer */}
                      <div className="disclaimer">
                        <FaInfoCircle />
                        <span><strong>Disclaimer:</strong> This AI analysis is for educational purposes only. Always consult qualified medical professionals for diagnosis.</span>
                      </div>
                    </div>
                  ) : result && !result.success ? (
                    <div className="error-result">
                      <FaTimesCircle className="error-icon" />
                      <h3>Analysis Failed</h3>
                      <p>{result.error}</p>
                    </div>
                  ) : (
                    <div className="no-results">
                      <FaBrain className="no-results-icon" />
                      <h3>Ready for Analysis</h3>
                      <p>Upload an MRI scan to begin AI-powered tumor detection</p>
                      <div className="no-results-features">
                        <span><FaCheckCircle /> 4 Tumor Types</span>
                        <span><FaCheckCircle /> 96.95% Accuracy</span>
                        <span><FaCheckCircle /> Instant Results</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Charts Section */}
            {result && result.success && (
              <div className="charts-section">
                <div className="panel chart-panel">
                  <div className="panel-header">
                    <h2><FaChartLine /> Probability Distribution</h2>
                    <span className="panel-subtitle">Bar chart showing all class probabilities</span>
                  </div>
                  <div className="chart-container bar-chart">
                    {getBarChartData() && <Bar data={getBarChartData()} options={chartOptions} />}
                  </div>
                </div>

                <div className="panel chart-panel">
                  <div className="panel-header">
                    <h2>Classification Breakdown</h2>
                    <span className="panel-subtitle">Proportional view of predictions</span>
                  </div>
                  <div className="chart-container doughnut-chart">
                    {getDoughnutData() && <Doughnut data={getDoughnutData()} options={doughnutOptions} />}
                  </div>
                </div>

                {history.length >= 2 && (
                  <div className="panel chart-panel wide">
                    <div className="panel-header">
                      <h2>Confidence Trend Analysis</h2>
                      <span className="panel-subtitle">Historical confidence tracking across scans</span>
                    </div>
                    <div className="chart-container line-chart">
                      {getLineChartData() && <Line data={getLineChartData()} options={lineChartOptions} />}
                    </div>
                  </div>
                )}

                {/* Detailed Probabilities Table */}
                <div className="panel chart-panel wide">
                  <div className="panel-header">
                    <h2>Detailed Analysis</h2>
                    <span className="panel-subtitle">Complete probability breakdown</span>
                  </div>
                  <div className="probabilities-table">
                    {Object.entries(result.all_probabilities)
                      .sort(([,a], [,b]) => b - a)
                      .map(([cls, prob], index) => (
                        <div 
                          key={cls} 
                          className={`prob-row ${cls === result.predicted_class ? 'highlighted' : ''}`}
                        >
                          <div className="prob-rank">#{index + 1}</div>
                          <div className="prob-name">{formatClassName(cls)}</div>
                          <div className="prob-bar-container">
                            <div 
                              className="prob-bar-fill"
                              style={{ 
                                width: `${prob}%`,
                                backgroundColor: cls === result.predicted_class 
                                  ? getTumorInfo(cls).color 
                                  : '#94a3b8'
                              }}
                            ></div>
                          </div>
                          <div className="prob-value">{prob.toFixed(2)}%</div>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            )}
          </>
        ) : (
          /* History Section */
          <div className="history-section">
            <div className="panel history-panel">
              <div className="panel-header">
                <h2><FaHistory /> Scan History</h2>
                <span className="panel-subtitle">Recent analysis records</span>
              </div>
              <div className="panel-content">
                {history.length > 0 ? (
                  <div className="history-grid">
                    {history.map((item) => (
                      <div key={item.id} className="history-card">
                        <img src={item.preview} alt="Scan" className="history-image" />
                        <div className="history-info">
                          <span 
                            className="history-diagnosis"
                            style={{ color: getTumorInfo(item.prediction).color }}
                          >
                            {formatClassName(item.prediction)}
                          </span>
                          <div className="history-meta">
                            <span className="history-confidence">{item.confidence.toFixed(1)}%</span>
                            <span 
                              className="history-severity"
                              style={{ backgroundColor: getTumorInfo(item.prediction).color }}
                            >
                              {getTumorInfo(item.prediction).severity}
                            </span>
                          </div>
                          <span className="history-time">{item.timestamp}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="no-history">
                    <FaHistory className="no-history-icon" />
                    <h3>No Scan History</h3>
                    <p>Analyzed scans will appear here</p>
                  </div>
                )}
              </div>
            </div>

            {/* History Stats Chart */}
            {history.length >= 2 && (
              <div className="panel chart-panel wide">
                <div className="panel-header">
                  <h2>Confidence Trend</h2>
                  <span className="panel-subtitle">Historical confidence analysis</span>
                </div>
                <div className="chart-container line-chart">
                  {getLineChartData() && <Line data={getLineChartData()} options={lineChartOptions} />}
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <div className="footer-brand">
            <FaBrain className="footer-icon" />
            <span>NeuroScan AI</span>
          </div>
          <p className="footer-text">
            Advanced Brain Tumor Detection System • Deep Learning Powered • Educational Project
          </p>
          <p className="footer-disclaimer">
            This tool is for educational purposes only and should not replace professional medical diagnosis.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
