# Water Quality Analysis Dashboard

A comprehensive Flask-based machine learning dashboard for water quality analysis with complete ML pipeline and Explainable AI (XAI) features.

## Features

### 📊 Dataset Overview
- Load and analyze water.csv dataset
- Display dataset statistics, missing values, and distributions
- Automatic sample dataset generation if water.csv doesn't exist

### ⚙️ Pre-processing Steps
- Handle missing values using mean imputation
- Label encoding for categorical variables
- Data splitting and feature scaling

### 🔧 Feature Engineering
- Create new features: pH_Hardness_Ratio, Solids_Conductivity_Ratio
- Generate Chemical_Index and Physical_Index
- Correlation analysis and visualization

### 🧠 Model Training
- Logistic Regression model with scikit-learn
- Real-time model training
- Model information and statistics

### 🔮 Prediction (Belief)
- Interactive water quality parameter input
- Real-time predictions with confidence scores
- Probability distributions for all classes

### ✅ User Input Validation
- Dual input validation for water quality and sewage parameters
- Classification results with accuracy
- Comparative analysis between inputs

### 📈 Model Performance Metrics
- Confusion Matrix visualization
- Accuracy, F1 Score, Precision, Recall
- Comprehensive classification report

### 🧠 Explainable AI (XAI)
- **SHAP Explanations**: Feature importance analysis
- **LIME Explanations**: Local interpretability
- **Counterfactual Explanations**: What-if scenarios

## Installation

1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Flask application:
```bash
python water_dashboard.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Use the dashboard tabs to navigate through different features:
   - **Dataset Overview**: Load and explore the dataset
   - **Preprocessing**: Clean and prepare data
   - **Feature Engineering**: Generate new features
   - **Model Training**: Train the Logistic Regression model
   - **Prediction**: Make predictions on custom inputs
   - **Validation**: Validate water quality and sewage inputs
   - **Metrics**: View model performance metrics
   - **XAI**: Generate explainable AI insights

## Dataset

The dashboard expects a `water.csv` file with the following columns:
- pH
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic_carbon
- Trihalomethanes
- Turbidity

If `water.csv` doesn't exist, the application automatically generates a sample dataset with 1000 records for demonstration purposes.

## Technical Stack

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **XAI Libraries**: SHAP, LIME
- **Frontend**: Bootstrap 5, JavaScript

## API Endpoints

- `GET /` - Main dashboard
- `GET /dataset_overview` - Dataset statistics
- `GET /preprocessing` - Data preprocessing
- `GET /feature_engineering` - Feature engineering
- `GET /train_model` - Model training
- `POST /predict` - Single prediction
- `POST /validate_input` - Dual input validation
- `GET /metrics` - Model performance metrics
- `GET /xai` - All XAI explanations
- `GET /shap` - SHAP explanations
- `GET /lime` - LIME explanations
- `GET /counterfactual` - Counterfactual explanations

## File Structure

```
Water/
├── water_dashboard.py          # Main Flask application
├── requirements.txt            # Python dependencies
├── README.md                  # This file
├── water.csv                  # Dataset (auto-generated if missing)
└── templates/
    └── dashboard.html         # Frontend template
```

## Features in Detail

### 1. Dataset Overview
- Displays dataset shape, columns, and basic statistics
- Shows first 5 rows and statistical summary
- Missing values analysis
- Distribution plots for key features

### 2. Pre-processing
- Missing value handling with mean imputation
- Label encoding for categorical variables
- Data splitting (80% train, 20% test)
- Standard scaling of features

### 3. Feature Engineering
- Creates ratio features (pH/Hardness, Solids/Conductivity)
- Generates composite indices (Chemical, Physical)
- Correlation matrix visualization

### 4. Model Training
- Logistic Regression with scikit-learn
- Real-time training on processed data
- Model configuration and statistics

### 5. Prediction System
- Interactive form for water quality parameters
- Real-time prediction with confidence scores
- Probability distribution across classes

### 6. Validation System
- Dual input forms for water quality and sewage
- Simultaneous classification of both inputs
- Comparative results with accuracy metrics

### 7. Performance Metrics
- Confusion matrix with visualization
- Accuracy, F1, Precision, Recall scores
- Detailed classification report

### 8. Explainable AI
- **SHAP**: Global feature importance
- **LIME**: Local prediction explanations
- **Counterfactual**: What-if analysis

## Notes

- The application automatically creates a sample dataset if `water.csv` is not found
- All visualizations are generated dynamically and embedded as base64 images
- The dashboard is fully responsive and works on different screen sizes
- Model training happens in real-time when requested

## Troubleshooting

1. **Port already in use**: Change the port in `water_dashboard.py` (line ending)
2. **Missing dependencies**: Ensure all packages from `requirements.txt` are installed
3. **Dataset issues**: The app will auto-generate a sample dataset if needed
4. **XAI errors**: Some XAI features may require additional setup for large datasets

## License

This project is for educational and demonstration purposes.
