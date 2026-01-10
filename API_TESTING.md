# API Testing Guide

## Using cURL

### 1. Health Check

```bash
curl http://localhost:5000/api/health
```

### 2. Get Available Classes

```bash
curl http://localhost:5000/api/classes
```

### 3. Predict Single Image

```bash
curl -X POST -F "file=@path/to/image.jpg" http://localhost:5000/api/predict
```

### 4. Get Statistics

```bash
curl http://localhost:5000/api/stats
```

### 5. Get Model Info

```bash
curl http://localhost:5000/api/model/info
```

## Using Python Requests

```python
import requests

# Upload and predict
url = "http://localhost:5000/api/predict"
files = {"file": open("path/to/image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Using Postman

1. Create new request
2. Method: POST
3. URL: `http://localhost:5000/api/predict`
4. Body: form-data
5. Key: `file` (type: File)
6. Value: Select your image
7. Send

## Expected Response

```json
{
  "success": true,
  "predicted_class": "glioma",
  "confidence": 96.5,
  "is_confident": true,
  "all_probabilities": {
    "glioma": 96.5,
    "meningioma": 2.1,
    "pituitary": 1.2,
    "no_tumor": 0.2
  },
  "top_predictions": [
    { "class": "glioma", "confidence": 96.5 },
    { "class": "meningioma", "confidence": 2.1 },
    { "class": "pituitary", "confidence": 1.2 }
  ],
  "warning": "Tumor detected: glioma. Please consult a medical professional.",
  "timestamp": "2026-01-06T10:30:00"
}
```

## Error Responses

### 400 - Bad Request

```json
{
  "success": false,
  "error": "No file provided"
}
```

### 500 - Server Error

```json
{
  "success": false,
  "error": "Model not loaded. Please train the model first."
}
```
