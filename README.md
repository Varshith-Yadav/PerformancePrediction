# LMS Performance Predictor

A comprehensive Learning Management System (LMS) performance prediction system using LSTM neural networks. This system monitors and predicts critical performance metrics to ensure zero downtime during critical assessment periods.

## Features

### What to Monitor:
- **Server Load Patterns**: CPU usage percentage with daily/weekly patterns and exam period spikes
- **Database Performance**: Query response times in milliseconds
- **API Response Times**: API endpoint latency tracking
- **Storage Capacity Trends**: Storage usage patterns and growth predictions

### Predictions:
- **Scaling Needs**: When system will need scaling based on threshold analysis
- **Peak Usage Times**: Identification of peak resource requirement periods
- **Crash Risk Assessment**: Potential crashes during exam periods with risk scoring

### Impact:
- **Zero Downtime**: Proactive scaling prevents downtime during critical assessment periods
- **Resource Optimization**: Predict peak usage times for better resource allocation
- **Risk Mitigation**: Early warning system for potential system failures

## Project Structure

```
performance_pred/
├── app.py                 # FastAPI application
├── model.ipynb           # Jupyter notebook for model training
├── model/
│   ├── train.py          # Model training script
│   └── predict.py        # Prediction functions
├── database/
│   └── connection.py     # Database connection utilities
├── models/               # Saved models (created after training)
│   ├── server_load_model.h5
│   ├── db_performance_model.h5
│   ├── api_response_time_model.h5
│   ├── storage_capacity_model.h5
│   └── scalers/          # Scaler files for each metric
└── requirements.txt      # Python dependencies
```

## Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Train Models** (Option 1 - Using Jupyter Notebook):
   - Open `model.ipynb` in Jupyter Notebook
   - Run all cells to generate data, train models, and save them as `.h5` files
   - Models will be saved in `models/` directory

3. **Train Models** (Option 2 - Using API):
```bash
# Start the FastAPI server
uvicorn app:app --reload

# In another terminal, train models via API
curl -X POST http://localhost:8000/train
```

## Usage

### Starting the FastAPI Server

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. **Root Endpoint**
```bash
GET http://localhost:8000/
```
Returns API information and available endpoints.

#### 2. **Train Models**
```bash
POST http://localhost:8000/train
```
Trains all LSTM models for the four performance metrics.

#### 3. **Get All Predictions**
```bash
GET http://localhost:8000/predict?hours=24
```
Returns predictions for all metrics for the next N hours (default: 24).

**Response**:
```json
{
  "predictions": {
    "server_load": [65.2, 68.5, ...],
    "db_performance": [120.3, 125.1, ...],
    "api_response_time": [85.2, 90.1, ...],
    "storage_capacity": [750.5, 752.3, ...]
  },
  "timestamps": ["2024-01-15T10:00:00", ...],
  "scaling_alerts": [...],
  "crash_risk": {...}
}
```

#### 4. **Scaling Analysis**
```bash
GET http://localhost:8000/scaling?hours=24
```
Analyzes when scaling is needed based on threshold analysis.

**Thresholds**:
- Server Load: > 80% (High: > 90%)
- Database Performance: > 200ms (High: > 300ms)
- API Response Time: > 150ms (High: > 200ms)
- Storage Capacity: > 900GB

#### 5. **Crash Risk Assessment**
```bash
GET http://localhost:8000/crash-risk?hours=24
```
Assesses crash risk during exam periods.

**Risk Factors**:
- Critical Server Load: > 90%
- Critical DB Performance: > 400ms
- Critical API Latency: > 250ms
- Critical Storage: > 950GB

**Risk Levels**:
- **Critical**: Risk score > 75% - Immediate scaling required
- **High**: Risk score > 50% - Monitor closely
- **Medium**: Risk score > 25% - Normal operations with caution
- **Low**: Risk score ≤ 25% - Normal operations

#### 6. **Get Specific Metric Prediction**
```bash
GET http://localhost:8000/metrics/server_load?hours=24
GET http://localhost:8000/metrics/db_performance?hours=24
GET http://localhost:8000/metrics/api_response_time?hours=24
GET http://localhost:8000/metrics/storage_capacity?hours=24
```

## Model Architecture

Each metric uses a multi-layer LSTM architecture:
- **Input Layer**: 24-hour sequence (lookback window)
- **LSTM Layer 1**: 64 units with return_sequences=True
- **Dropout**: 0.2
- **LSTM Layer 2**: 32 units with return_sequences=True
- **Dropout**: 0.2
- **LSTM Layer 3**: 16 units
- **Dropout**: 0.1
- **Output Layer**: Dense(1) with linear activation

## Data Generation

The system can work with:
1. **Real Database Data**: Connect to MySQL database via `database/connection.py`
2. **Synthetic Data**: Automatically generates realistic LMS performance data if database is unavailable

Synthetic data includes:
- Daily and weekly patterns
- Exam period spikes (every 2 weeks, 3 days duration)
- Correlated metrics (DB and API correlate with server load)
- Storage growth trends

## Example Usage

### Python Client Example

```python
import requests

# Get predictions
response = requests.get("http://localhost:8000/predict?hours=48")
data = response.json()

print(f"Server Load - Max: {max(data['predictions']['server_load']):.2f}%")
print(f"Crash Risk Level: {data['crash_risk']['risk_level']}")

# Check scaling needs
scaling = requests.get("http://localhost:8000/scaling?hours=24").json()
if scaling['scaling_alerts']:
    print("⚠️ Scaling required!")
    for alert in scaling['scaling_alerts']:
        print(f"  - {alert['metric']}: {alert['severity']} severity")
```

### cURL Examples

```bash
# Get 24-hour predictions
curl http://localhost:8000/predict?hours=24

# Check scaling needs
curl http://localhost:8000/scaling

# Assess crash risk
curl http://localhost:8000/crash-risk

# Get specific metric
curl http://localhost:8000/metrics/server_load?hours=48
```

## Monitoring Dashboard

You can integrate this API with monitoring dashboards like:
- Grafana
- Prometheus
- Custom web dashboards

## Database Configuration

Update `database/connection.py` with your MySQL credentials:

```python
def get_connection():
    return mysql.connector.connect(
        host='localhost',
        user='your_user',
        password='your_password',
        database='your_database'
    )
```

The database should have a table with columns:
- `timestamp`: DateTime
- `server_load`: Float (CPU usage %)
- `db_performance`: Float (response time in ms)
- `api_response_time`: Float (response time in ms)
- `storage_capacity`: Float (storage used in GB)

## Model Retraining

Models should be retrained periodically (e.g., weekly) to adapt to changing patterns:

```bash
# Via API
curl -X POST http://localhost:8000/train

# Or run the notebook again
# Or use: python -c "from model.train import train_model; train_model()"
```

## Performance Thresholds

### Scaling Thresholds:
- **Server Load**: 80% (Warning), 90% (Critical)
- **Database**: 200ms (Warning), 300ms (Critical)
- **API**: 150ms (Warning), 200ms (Critical)
- **Storage**: 900GB (Warning), 950GB (Critical)

## License

This project is for educational and production use.

## Support

For issues or questions, please check the API documentation at:
`http://localhost:8000/docs` (Swagger UI)
`http://localhost:8000/redoc` (ReDoc)

