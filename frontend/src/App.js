import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './App.css';

const API_URL = 'http://localhost:5000';

function App() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [viewMode, setViewMode] = useState('original'); // original, overlay, heatmap
    const [systemStatus, setSystemStatus] = useState('ready');
    const [processingSteps, setProcessingSteps] = useState({
        loaded: false,
        preprocessing: false,
        analysis: false,
        report: false
    });

    // Check API health on mount
    React.useEffect(() => {
        checkSystemHealth();
    }, []);

    const checkSystemHealth = async () => {
        try {
            const response = await axios.get(`${API_URL}/api/health`);
            setSystemStatus(response.data.status === 'healthy' ? 'ready' : 'offline');
        } catch (error) {
            setSystemStatus('offline');
        }
    };

    const onDrop = useCallback((acceptedFiles) => {
        const file = acceptedFiles[0];
        if (file) {
            setSelectedFile(file);
            setResult(null);
            setProcessingSteps({
                loaded: true,
                preprocessing: false,
                analysis: false,
                report: false
            });

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
            alert('Please load an MRI scan first');
            return;
        }

        setLoading(true);
        setResult(null);

        // Update processing steps
        setProcessingSteps({
            loaded: true,
            preprocessing: true,
            analysis: false,
            report: false
        });

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await axios.post(`${API_URL}/api/predict`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            setResult(response.data);

            // Complete all steps
            setProcessingSteps({
                loaded: true,
                preprocessing: true,
                analysis: true,
                report: true
            });
        } catch (error) {
            console.error('Error:', error);
            setResult({
                success: false,
                error: error.response?.data?.error || 'Analysis failed. Please verify the image format and try again.'
            });

            setProcessingSteps({
                loaded: true,
                preprocessing: true,
                analysis: false,
                report: false
            });
        } finally {
            setLoading(false);
        }
    };

    const handleClear = () => {
        setSelectedFile(null);
        setPreview(null);
        setResult(null);
        setProcessingSteps({
            loaded: false,
            preprocessing: false,
            analysis: false,
            report: false
        });
    };

    const formatClassName = (className) => {
        return className.split('_').map(word =>
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    };

    const getRiskLevel = (className, confidence) => {
        if (className === 'no_tumor') return 'Normal';
        if (confidence >= 80) return 'High Confidence';
        if (confidence >= 60) return 'Moderate Confidence';
        return 'Low Confidence';
    };

    const getStatusIcon = (step) => {
        if (!processingSteps[step]) return '○';
        if (loading && step === 'preprocessing') return '⏳';
        if (loading && step === 'analysis') return '⏳';
        if (loading && step === 'report') return '⏳';
        return '✓';
    };

    return (
        <div className="clinical-app">
            {/* Clinical Header Bar */}
            <header className="clinical-header">
                <div className="header-left">
                    <div className="system-logo">
                        <span className="brain-icon">🧠</span>
                        <div className="system-title">
                            <h1>NeuroVision AI</h1>
                            <span className="subtitle">MRI Tumor Analysis Assistant</span>
                        </div>
                    </div>
                </div>
                <div className="header-right">
                    <div className="system-info">
                        <div className="info-item">
                            <span className="info-label">System Version:</span>
                            <span className="info-value">v1.0 Clinical Prototype</span>
                        </div>
                        <div className="info-item">
                            <span className="info-label">Model:</span>
                            <span className="info-value">EfficientNet-B4 Based CNN</span>
                        </div>
                        <div className="info-item">
                            <span className="info-label">Status:</span>
                            <span className={`status-indicator ${systemStatus}`}>
                                {systemStatus === 'ready' ? '● System Ready' : '● System Offline'}
                            </span>
                        </div>
                    </div>
                </div>
            </header>

            {/* Main Clinical Dashboard */}
            <div className="clinical-dashboard">
                {/* Left Panel - Study Information */}
                <div className="study-panel">
                    <div className="panel-section">
                        <h2 className="panel-title">Study Information</h2>

                        {/* Upload Section */}
                        <div className="section-block">
                            <h3 className="section-subtitle">Upload MRI Scan</h3>
                            {!preview ? (
                                <div
                                    {...getRootProps()}
                                    className={`clinical-dropzone ${isDragActive ? 'active' : ''}`}
                                >
                                    <input {...getInputProps()} />
                                    <div className="dropzone-content">
                                        <div className="upload-icon">📁</div>
                                        <p className="dropzone-text">Drag & Drop MRI Image</p>
                                        <button className="btn-clinical">Browse Files</button>
                                        <p className="file-types">PNG, JPG, JPEG, BMP, TIFF</p>
                                    </div>
                                </div>
                            ) : (
                                <div className="file-loaded">
                                    <div className="file-info-row">
                                        <span className="label">Filename:</span>
                                        <span className="value">{selectedFile?.name}</span>
                                    </div>
                                    <div className="file-info-row">
                                        <span className="label">Size:</span>
                                        <span className="value">{(selectedFile?.size / 1024).toFixed(2)} KB</span>
                                    </div>
                                    <div className="file-info-row">
                                        <span className="label">Type:</span>
                                        <span className="value">{selectedFile?.type || 'Image'}</span>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Scan Metadata */}
                        <div className="section-block">
                            <h3 className="section-subtitle">Scan Metadata</h3>
                            <div className="metadata-grid">
                                <div className="metadata-item">
                                    <span className="meta-label">Study ID:</span>
                                    <span className="meta-value">{selectedFile ? 'MRI-' + Date.now().toString().slice(-6) : '—'}</span>
                                </div>
                                <div className="metadata-item">
                                    <span className="meta-label">Scan Type:</span>
                                    <span className="meta-value">Brain MRI</span>
                                </div>
                                <div className="metadata-item">
                                    <span className="meta-label">Date:</span>
                                    <span className="meta-value">{new Date().toLocaleDateString()}</span>
                                </div>
                                <div className="metadata-item">
                                    <span className="meta-label">Image Dimensions:</span>
                                    <span className="meta-value">{selectedFile ? 'Auto-detected' : '—'}</span>
                                </div>
                            </div>
                        </div>

                        {/* Processing Status */}
                        <div className="section-block">
                            <h3 className="section-subtitle">Processing Status</h3>
                            <div className="status-steps">
                                <div className={`status-step ${processingSteps.loaded ? 'completed' : ''}`}>
                                    <span className="step-icon">{getStatusIcon('loaded')}</span>
                                    <span>Image Loaded</span>
                                </div>
                                <div className={`status-step ${processingSteps.preprocessing ? 'completed' : ''} ${loading && processingSteps.loaded ? 'active' : ''}`}>
                                    <span className="step-icon">{getStatusIcon('preprocessing')}</span>
                                    <span>Preprocessing</span>
                                </div>
                                <div className={`status-step ${processingSteps.analysis ? 'completed' : ''} ${loading && processingSteps.preprocessing ? 'active' : ''}`}>
                                    <span className="step-icon">{getStatusIcon('analysis')}</span>
                                    <span>AI Analysis</span>
                                </div>
                                <div className={`status-step ${processingSteps.report ? 'completed' : ''} ${loading && processingSteps.analysis ? 'active' : ''}`}>
                                    <span className="step-icon">{getStatusIcon('report')}</span>
                                    <span>Report Generation</span>
                                </div>
                            </div>
                        </div>

                        {/* Action Buttons */}
                        <div className="action-buttons">
                            <button
                                className="btn-primary-clinical"
                                onClick={handlePredict}
                                disabled={!selectedFile || loading}
                            >
                                {loading ? 'Analyzing...' : 'Start Analysis'}
                            </button>
                            <button
                                className="btn-secondary-clinical"
                                onClick={handleClear}
                                disabled={loading}
                            >
                                Clear Study
                            </button>
                        </div>
                    </div>
                </div>

                {/* Right Panel - MRI Viewer & Analysis */}
                <div className="viewer-panel">
                    {/* MRI Viewer */}
                    <div className="panel-section">
                        <div className="section-header">
                            <h2 className="panel-title">MRI Scan & AI Visualization</h2>
                            {preview && (
                                <div className="view-controls">
                                    <button
                                        className={`view-btn ${viewMode === 'original' ? 'active' : ''}`}
                                        onClick={() => setViewMode('original')}
                                    >
                                        Original
                                    </button>
                                    <button
                                        className={`view-btn ${viewMode === 'overlay' ? 'active' : ''}`}
                                        onClick={() => setViewMode('overlay')}
                                        disabled={!result?.success}
                                    >
                                        AI Overlay
                                    </button>
                                    <button
                                        className={`view-btn ${viewMode === 'heatmap' ? 'active' : ''}`}
                                        onClick={() => setViewMode('heatmap')}
                                        disabled={!result?.success}
                                    >
                                        Heatmap
                                    </button>
                                </div>
                            )}
                        </div>

                        <div className="mri-viewer">
                            {!preview ? (
                                <div className="viewer-placeholder">
                                    <div className="placeholder-icon">🖼️</div>
                                    <p>No MRI scan loaded</p>
                                    <p className="placeholder-hint">Upload an image to begin analysis</p>
                                </div>
                            ) : (
                                <div className="mri-display">
                                    <img src={preview} alt="MRI Scan" className="mri-image" />
                                    {result?.success && viewMode === 'overlay' && (
                                        <div className="ai-overlay">
                                            <div className="overlay-label">AI Detection Overlay</div>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>

                        {result?.success && (
                            <div className="confidence-indicator">
                                <div className="confidence-label">Prediction Confidence</div>
                                <div className="confidence-bar-wrapper">
                                    <div
                                        className="confidence-bar-fill"
                                        style={{ width: `${result.confidence}%` }}
                                    >
                                        <span className="confidence-text">{result.confidence}%</span>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* AI Findings Panel */}
                    {result && (
                        <div className="panel-section findings-panel">
                            <h2 className="panel-title">AI Analysis Summary</h2>

                            {result.success ? (
                                <>
                                    {/* Detection Outcome */}
                                    <div className="findings-section">
                                        <h3 className="findings-subtitle">Detection Outcome</h3>
                                        <div className="findings-grid">
                                            <div className="finding-item">
                                                <span className="finding-label">Predicted Class:</span>
                                                <span className="finding-value highlight">{formatClassName(result.predicted_class)}</span>
                                            </div>
                                            <div className="finding-item">
                                                <span className="finding-label">Abnormality Detected:</span>
                                                <span className="finding-value">{result.predicted_class === 'no_tumor' ? 'No' : 'Yes'}</span>
                                            </div>
                                            <div className="finding-item">
                                                <span className="finding-label">Confidence Score:</span>
                                                <span className="finding-value">{result.confidence}%</span>
                                            </div>
                                            <div className="finding-item">
                                                <span className="finding-label">Risk Indicator:</span>
                                                <span className={`finding-value risk-${getRiskLevel(result.predicted_class, result.confidence).toLowerCase().replace(' ', '-')}`}>
                                                    {getRiskLevel(result.predicted_class, result.confidence)}
                                                </span>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Class Probabilities */}
                                    <div className="findings-section">
                                        <h3 className="findings-subtitle">Class Probability Distribution</h3>
                                        <div className="probability-chart">
                                            {Object.entries(result.all_probabilities).map(([className, probability]) => (
                                                <div key={className} className="probability-row">
                                                    <span className="prob-label">{formatClassName(className)}</span>
                                                    <div className="prob-bar-container">
                                                        <div
                                                            className="prob-bar"
                                                            style={{ width: `${probability}%` }}
                                                        ></div>
                                                        <span className="prob-value">{probability.toFixed(1)}%</span>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Medical Report */}
                                    <div className="findings-section">
                                        <h3 className="findings-subtitle">Preliminary AI Report</h3>
                                        <div className="medical-report">
                                            <p className="report-text">
                                                {result.predicted_class === 'no_tumor'
                                                    ? `AI-based analysis suggests no significant intracranial lesion detected. The scan appears within normal parameters based on the algorithmic assessment.`
                                                    : `AI-based analysis suggests the presence of a suspicious intracranial lesion consistent with ${formatClassName(result.predicted_class)}.`
                                                }
                                            </p>
                                            <p className="report-text">
                                                Model confidence: {result.confidence}%
                                            </p>
                                            <p className="report-disclaimer">
                                                This output is intended as a clinical decision support tool and must be reviewed by a qualified radiologist.
                                            </p>
                                        </div>
                                        <div className="report-actions">
                                            <button className="btn-report" disabled>
                                                📄 Generate PDF Report
                                            </button>
                                            <button className="btn-report" disabled>
                                                💾 Export Findings
                                            </button>
                                        </div>
                                    </div>
                                </>
                            ) : (
                                <div className="error-message">
                                    <div className="error-icon">⚠️</div>
                                    <p className="error-text">Analysis Failed</p>
                                    <p className="error-detail">{result.error}</p>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>

            {/* Medical Disclaimer */}
            <div className="clinical-disclaimer">
                <div className="disclaimer-content">
                    <div className="disclaimer-header">
                        <span className="disclaimer-icon">⚠️</span>
                        <h3>Clinical Decision Support Only</h3>
                    </div>
                    <p className="disclaimer-text">
                        This system does not provide a medical diagnosis. Results are algorithmic predictions based on
                        deep learning models and must be reviewed by a licensed medical professional. This tool is
                        intended for research and clinical decision support purposes only. Do not use this system as
                        the sole basis for diagnostic or treatment decisions.
                    </p>
                </div>
            </div>

            {/* Footer */}
            <footer className="clinical-footer">
                <div className="footer-content">
                    <span>NeuroVision AI Clinical Prototype v1.0</span>
                    <span>•</span>
                    <span>EfficientNet-B4 Deep Learning Model</span>
                    <span>•</span>
                    <span>For Research & Educational Use</span>
                </div>
            </footer>
        </div>
    );
}

export default App;
