import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './App.css';
import {
    FaBrain,
    FaUpload,
    FaTrash,
    FaCheckCircle,
    FaExclamationTriangle,
    FaChartBar,
    FaCog,
    FaShieldAlt,
    FaRocket
} from 'react-icons/fa';

const API_URL = 'http://localhost:5000';

function App() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [stats, setStats] = useState(null);

    // Load stats on mount
    React.useEffect(() => {
        fetchStats();
    }, []);

    const fetchStats = async () => {
        try {
            const response = await axios.get(`${API_URL}/api/stats`);
            setStats(response.data);
        } catch (error) {
            console.error('Error fetching stats:', error);
        }
    };

    const onDrop = useCallback((acceptedFiles) => {
        const file = acceptedFiles[0];
        if (file) {
            setSelectedFile(file);
            setResult(null);

            // Create preview
            const reader = new FileReader();
            reader.onloadend = () => {
                setPreview(reader.result);
            };
            reader.readAsDataURL(file);
        }
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'image/*': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        },
        multiple: false
    });

    const handlePredict = async () => {
        if (!selectedFile) {
            alert('Please select an image first');
            return;
        }

        setLoading(true);
        setResult(null);

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await axios.post(`${API_URL}/api/predict`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            setResult(response.data);
            fetchStats();
        } catch (error) {
            console.error('Error:', error);
            setResult({
                success: false,
                error: error.response?.data?.error || 'Failed to make prediction. Please try again.'
            });
        } finally {
            setLoading(false);
        }
    };

    const handleClear = () => {
        setSelectedFile(null);
        setPreview(null);
        setResult(null);
    };

    const getConfidenceClass = (confidence) => {
        if (confidence >= 80) return 'confidence-high';
        if (confidence >= 60) return 'confidence-medium';
        return 'confidence-low';
    };

    const formatClassName = (className) => {
        return className.split('_').map(word =>
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    };

    return (
        <div className="App">
            <div className="container">
                {/* Header */}
                <header className="header">
                    <h1>
                        <FaBrain style={{ display: 'inline', marginRight: '15px' }} />
                        Brain Tumor Detection
                    </h1>
                    <p>Advanced AI-Powered MRI Analysis System</p>
                    <p className="subtitle">
                        Deep Learning Model with EfficientNet-B4 Architecture
                    </p>
                </header>

                {/* Statistics */}
                {stats && (
                    <div className="stats-grid fade-in">
                        <div className="stat-card">
                            <div className="stat-value">{stats.total_predictions}</div>
                            <div className="stat-label">Total Predictions</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-value">
                                {stats.class_distribution ? Object.keys(stats.class_distribution).length : 0}
                            </div>
                            <div className="stat-label">Classes Detected</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-value">95%+</div>
                            <div className="stat-label">Model Accuracy</div>
                        </div>
                    </div>
                )}

                {/* Main Content */}
                <div className="main-content">
                    {/* Upload Section */}
                    <div className="card upload-section">
                        <h2>
                            <FaUpload />
                            Upload MRI Image
                        </h2>

                        {!preview ? (
                            <div
                                {...getRootProps()}
                                className={`dropzone ${isDragActive ? 'active' : ''}`}
                            >
                                <input {...getInputProps()} />
                                <div className="dropzone-icon">
                                    <FaUpload />
                                </div>
                                <h3>Drop MRI image here</h3>
                                <p>or click to select from your computer</p>
                                <p style={{ marginTop: '10px', fontSize: '0.9rem', color: '#999' }}>
                                    Supported formats: PNG, JPG, JPEG, BMP, TIFF
                                </p>
                            </div>
                        ) : (
                            <div className="image-preview">
                                <img src={preview} alt="Preview" className="preview-image" />
                                <div className="image-info">
                                    <p><strong>Filename:</strong> {selectedFile?.name}</p>
                                    <p><strong>Size:</strong> {(selectedFile?.size / 1024).toFixed(2)} KB</p>
                                    <p><strong>Type:</strong> {selectedFile?.type}</p>
                                </div>
                                <div className="button-group">
                                    <button
                                        className="btn btn-primary"
                                        onClick={handlePredict}
                                        disabled={loading}
                                    >
                                        {loading ? (
                                            <>
                                                <div className="spinner" style={{ width: '20px', height: '20px', margin: 0 }}></div>
                                                Analyzing...
                                            </>
                                        ) : (
                                            <>
                                                <FaBrain />
                                                Analyze Image
                                            </>
                                        )}
                                    </button>
                                    <button
                                        className="btn btn-secondary"
                                        onClick={handleClear}
                                        disabled={loading}
                                    >
                                        <FaTrash />
                                        Clear
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Results Section */}
                    <div className="card results-section">
                        <h2>
                            <FaChartBar />
                            Analysis Results
                        </h2>

                        {loading && (
                            <div className="loading">
                                <div className="spinner"></div>
                                <p>Analyzing MRI image using deep learning model...</p>
                            </div>
                        )}

                        {!loading && !result && (
                            <div style={{ textAlign: 'center', padding: '60px 20px', color: '#999' }}>
                                <FaChartBar style={{ fontSize: '4rem', marginBottom: '20px', opacity: 0.3 }} />
                                <p>Upload an MRI image to see analysis results</p>
                            </div>
                        )}

                        {!loading && result && (
                            <>
                                {result.success ? (
                                    <div>
                                        <div className="result-header">
                                            <FaCheckCircle style={{ fontSize: '3rem', color: '#28a745', marginBottom: '15px' }} />
                                            <h3>Detection Complete</h3>
                                            <div className="predicted-class">
                                                {formatClassName(result.predicted_class)}
                                            </div>
                                            <div className={`confidence-badge ${getConfidenceClass(result.confidence)}`}>
                                                Confidence: {result.confidence}%
                                            </div>
                                        </div>

                                        {result.warning && (
                                            <div className="warning-box">
                                                <FaExclamationTriangle style={{ marginRight: '10px' }} />
                                                <strong>Medical Attention Required</strong>
                                                <p>{result.warning}</p>
                                            </div>
                                        )}

                                        <div className="probabilities">
                                            <h3 style={{ marginBottom: '20px' }}>Classification Probabilities</h3>
                                            {Object.entries(result.all_probabilities).map(([className, probability]) => (
                                                <div key={className} className="probability-item">
                                                    <div className="probability-label">
                                                        {formatClassName(className)}
                                                    </div>
                                                    <div className="probability-bar-container">
                                                        <div
                                                            className="probability-bar"
                                                            style={{ width: `${probability}%` }}
                                                        >
                                                            {probability.toFixed(1)}%
                                                        </div>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>

                                        <div style={{ marginTop: '30px', padding: '20px', background: '#f8f9fa', borderRadius: '10px' }}>
                                            <h4 style={{ marginBottom: '15px' }}>Top Predictions</h4>
                                            {result.top_predictions.map((pred, index) => (
                                                <div key={index} style={{ marginBottom: '10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                                    <span style={{ fontWeight: '600' }}>
                                                        {index + 1}. {formatClassName(pred.class)}
                                                    </span>
                                                    <span style={{ color: '#667eea', fontWeight: '600' }}>
                                                        {pred.confidence.toFixed(2)}%
                                                    </span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                ) : (
                                    <div style={{ textAlign: 'center', padding: '40px', color: '#dc3545' }}>
                                        <FaExclamationTriangle style={{ fontSize: '3rem', marginBottom: '20px' }} />
                                        <h3>Analysis Failed</h3>
                                        <p>{result.error}</p>
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                </div>

                {/* About Section */}
                <div className="about-section">
                    <h2 style={{ textAlign: 'center', fontSize: '2rem', marginBottom: '15px' }}>
                        System Features
                    </h2>
                    <p style={{ textAlign: 'center', color: '#666', marginBottom: '30px' }}>
                        State-of-the-art deep learning technology for accurate brain tumor detection
                    </p>

                    <div className="about-grid">
                        <div className="feature-card">
                            <div className="feature-icon">
                                <FaRocket />
                            </div>
                            <h3>High Accuracy</h3>
                            <p>
                                Utilizes EfficientNet-B4 architecture with transfer learning,
                                achieving over 95% accuracy on test datasets.
                            </p>
                        </div>

                        <div className="feature-card">
                            <div className="feature-icon">
                                <FaBrain />
                            </div>
                            <h3>Multi-Class Detection</h3>
                            <p>
                                Capable of detecting and classifying multiple tumor types:
                                Glioma, Meningioma, Pituitary tumors, and healthy tissue.
                            </p>
                        </div>

                        <div className="feature-card">
                            <div className="feature-icon">
                                <FaCog />
                            </div>
                            <h3>Advanced Preprocessing</h3>
                            <p>
                                Implements CLAHE enhancement and denoising techniques
                                for optimal image quality and analysis.
                            </p>
                        </div>

                        <div className="feature-card">
                            <div className="feature-icon">
                                <FaShieldAlt />
                            </div>
                            <h3>Reliable Results</h3>
                            <p>
                                Provides confidence scores and multiple predictions
                                to assist medical professionals in diagnosis.
                            </p>
                        </div>
                    </div>

                    <div style={{ marginTop: '40px', padding: '30px', background: '#fff3cd', borderRadius: '15px' }}>
                        <h3 style={{ color: '#856404', marginBottom: '15px' }}>
                            <FaExclamationTriangle style={{ marginRight: '10px' }} />
                            Important Medical Disclaimer
                        </h3>
                        <p style={{ color: '#856404', lineHeight: '1.8' }}>
                            This system is designed as a diagnostic aid and should not replace professional medical judgment.
                            All results should be reviewed and validated by qualified medical professionals.
                            If you suspect a medical condition, please consult with a healthcare provider immediately.
                        </p>
                    </div>
                </div>

                {/* Footer */}
                <footer className="footer">
                    <p style={{ fontSize: '1.1rem', fontWeight: '600' }}>
                        Advanced Brain Tumor Detection System
                    </p>
                    <p>Powered by Deep Learning & Artificial Intelligence</p>
                    <p style={{ fontSize: '0.9rem', marginTop: '10px' }}>
                        &copy; 2026 | Built with TensorFlow, Keras & React
                    </p>
                </footer>
            </div>
        </div>
    );
}

export default App;
