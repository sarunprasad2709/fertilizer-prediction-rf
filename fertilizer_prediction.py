# Install required packages
!pip install -q kaggle pandas numpy scikit-learn matplotlib seaborn

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import warnings
import os
import zipfile
from google.colab import files
from google.colab import drive
import json

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

# Method 1: Upload kaggle.json file
print("Please upload your kaggle.json file when prompted...")
try:
    uploaded = files.upload()
    
    # Move kaggle.json to the right location
    os.makedirs('/root/.kaggle', exist_ok=True)
    
    for filename in uploaded.keys():
        if filename == 'kaggle.json':
            os.rename(filename, '/root/.kaggle/kaggle.json')
            os.chmod('/root/.kaggle/kaggle.json', 0o600)
            print(" Kaggle credentials set up successfully!")
            break
    else:
        print(" kaggle.json not found in uploaded files")
        
except Exception as e:
    print(f" Error setting up Kaggle credentials: {e}")
    print("\n Alternative method:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Click 'Create New API Token'")
    print("3. Upload the downloaded kaggle.json file when prompted above")

print("\n Downloading dataset from Kaggle")
print("=" * 60)

# Download the competition dataset
try:
    # Download competition files
    !kaggle competitions download -c playground-series-s5e6
    
    # Extract the files
    with zipfile.ZipFile('playground-series-s5e6.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
    
    # List downloaded files
    print("\n Downloaded files:")
    for file in os.listdir('.'):
        if file.endswith('.csv'):
            size = os.path.getsize(file) / (1024*1024)  # Size in MB
            print(f"  • {file} ({size:.1f} MB)")
    
    print(" Dataset downloaded successfully!")
    
except Exception as e:
    print(f" Error downloading dataset: {e}")

def load_and_explore_data():
    """Load and explore the dataset"""
    try:
        # Load datasets
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        sample_submission = pd.read_csv('sample_submission.csv')
        
        print(f" Dataset Shapes:")
        print(f" Training data: {train_df.shape}")
        print(f" Test data: {test_df.shape}")
        print(f" Sample submission: {sample_submission.shape}")
        
        print(f"\n Training Data Info:")
        print(f"  Columns: {list(train_df.columns)}")
        print(f"  Data types:")
        for col in train_df.columns:
            print(f"    - {col}: {train_df[col].dtype}")
        
        print(f"\n  Target Variable Analysis:")
        target_counts = train_df['Fertilizer Name'].value_counts()
        print(f"  Unique fertilizers: {len(target_counts)}")
        print(f"  Most common fertilizers:")
        for i, (fertilizer, count) in enumerate(target_counts.head(10).items()):
            print(f"    {i+1}. {fertilizer}: {count} samples")
        
        # Display sample data
        print(f"\n Sample Training Data:")
        print(train_df.head())
        
        print(f"\n Sample Test Data:")
        print(test_df.head())
        
        print(f"\n Sample Submission Format:")
        print(sample_submission.head())
        
        return train_df, test_df, sample_submission
        
    except Exception as e:
        print(f" Error loading data: {e}")
        return None, None, None

# Load the data
train_df, test_df, sample_submission = load_and_explore_data()


if train_df is not None:
    print("\n Data Visualization")
    print("=" * 60)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Target distribution
    target_counts = train_df['Fertilizer Name'].value_counts().head(15)
    axes[0, 0].bar(range(len(target_counts)), target_counts.values)
    axes[0, 0].set_title('Top 15 Fertilizer Types Distribution')
    axes[0, 0].set_xlabel('Fertilizer Type (Index)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Missing values heatmap
    missing_data = train_df.isnull().sum()
    if missing_data.sum() > 0:
        axes[0, 1].bar(missing_data.index, missing_data.values)
        axes[0, 1].set_title('Missing Values by Column')
        axes[0, 1].tick_params(axis='x', rotation=45)
    else:
        axes[0, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                       transform=axes[0, 1].transAxes, fontsize=14)
        axes[0, 1].set_title('Missing Values Analysis')
    
    # 3. Data types distribution
    dtype_counts = train_df.dtypes.value_counts()
    axes[1, 0].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
    axes[1, 0].set_title('Data Types Distribution')
    
    # 4. Dataset size comparison
    sizes = [len(train_df), len(test_df)]
    labels = ['Training Set', 'Test Set']
    axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Dataset Size Comparison')
    
    plt.tight_layout()
    plt.show()

def preprocess_data(train_df, test_df):
    """
    Comprehensive data preprocessing with one-hot encoding
    """
    print("🔄 Starting data preprocessing...")
    
    # Separate features and target
    if 'Fertilizer Name' in train_df.columns:
        X_train = train_df.drop(['Fertilizer Name'], axis=1)
        y_train = train_df['Fertilizer Name']
    else:
        X_train = train_df.copy()
        y_train = None
    
    X_test = test_df.copy()
    
    print(f"  • Training features shape: {X_train.shape}")
    print(f"  • Test features shape: {X_test.shape}")
    
    # Handle missing values
    print("  • Handling missing values...")
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            # Categorical: fill with mode
            mode_val = X_train[col].mode().iloc[0] if not X_train[col].mode().empty else 'Unknown'
            X_train[col] = X_train[col].fillna(mode_val)
            X_test[col] = X_test[col].fillna(mode_val)
        else:
            # Numerical: fill with mean
            mean_val = X_train[col].mean()
            X_train[col] = X_train[col].fillna(mean_val)
            X_test[col] = X_test[col].fillna(mean_val)
    
    # Identify feature types
    categorical_features = []
    numerical_features = []
    
    for col in X_train.columns:
        if col == 'id':
            continue  # Skip ID column
        elif X_train[col].dtype == 'object':
            categorical_features.append(col)
        else:
            numerical_features.append(col)
    
    print(f"  Categorical features ({len(categorical_features)}): {categorical_features}")
    print(f"  Numerical features ({len(numerical_features)}): {numerical_features}")
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'  # Drop any remaining columns (like 'id')
    )
    
    print(" Preprocessing pipeline created successfully!")
    
    return X_train, X_test, y_train, preprocessor, categorical_features, numerical_features

def train_random_forest_model(X_train, y_train, preprocessor):
    print(" Training Random Forest model...")
    
    # Create complete pipeline
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
        ))
    ])
    
    # Split for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"  Training set: {X_train_split.shape[0]} samples")
    print(f"  Validation set: {X_val_split.shape[0]} samples")
    
    # Train model
    print("  Fitting Random Forest...")
    rf_pipeline.fit(X_train_split, y_train_split)
    
    # Validate model
    print("  Evaluating on validation set...")
    val_predictions = rf_pipeline.predict(X_val_split)
    val_accuracy = accuracy_score(y_val_split, val_predictions)
    
    print(f"  Validation Accuracy: {val_accuracy:.4f}")
    
    # Calculate MAP@3 on validation set
    val_map3 = calculate_map3_validation(rf_pipeline, X_val_split, y_val_split)
    print(f"  Validation MAP@3: {val_map3:.4f}")
    
    # Train on full dataset
    print("  Training on full dataset...")
    rf_pipeline.fit(X_train, y_train)
    
    return rf_pipeline

def calculate_map3_validation(model, X_val, y_val):
    try:
        pred_proba = model.predict_proba(X_val)
        classes = model.classes_
        
        total_ap = 0
        for i, true_label in enumerate(y_val):
            # Get top 3 predictions
            top3_indices = np.argsort(pred_proba[i])[-3:][::-1]
            top3_predictions = [classes[idx] for idx in top3_indices]
            
            # Calculate AP for this sample
            ap = 0
            for j, pred in enumerate(top3_predictions):
                if pred == true_label:
                    ap = 1.0 / (j + 1)
                    break
            
            total_ap += ap
        
        return total_ap / len(y_val)
    except:
        return 0.0

def generate_top3_predictions(model, X_test, test_df):
    # Get prediction probabilities
    print("  Computing prediction probabilities...")
    pred_proba = model.predict_proba(X_test)
    classes = model.classes_
    
    print(f"  Number of classes: {len(classes)}")
    print(f"  Test samples: {len(pred_proba)}")
    
    # Generate top 3 predictions
    predictions = []
    print("  Generating top-3 predictions...")
    
    for i in range(len(pred_proba)):
        # Get indices of top 3 probabilities
        top3_indices = np.argsort(pred_proba[i])[-3:][::-1]
        
        # Get corresponding class names
        top3_classes = [classes[idx] for idx in top3_indices]
        
        # Join with spaces as required by submission format
        prediction_string = ' '.join(top3_classes)
        predictions.append(prediction_string)
        
        # Progress indicator
        if (i + 1) % 10000 == 0:
            print(f"    Processed {i + 1}/{len(pred_proba)} samples")
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'Fertilizer Name': predictions
    })
    
    return submission_df

def calculate_map3(y_true, y_pred_top3):
    
    def average_precision_k(actual, predicted, k=3):
        if len(predicted) > k:
            predicted = predicted[:k]
        
        score = 0.0
        num_hits = 0.0
        
        for i, p in enumerate(predicted):
            if p == actual:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
                break  # Stop after first correct prediction
        
        return score
    
    # Calculate AP for each sample and take mean
    ap_scores = []
    for true_label, pred_string in zip(y_true, y_pred_top3):
        predicted_labels = pred_string.split()
        ap = average_precision_k(true_label, predicted_labels, k=3)
        ap_scores.append(ap)
    
    return np.mean(ap_scores)

if train_df is not None and test_df is not None:
    print("\n Starting model training pipeline")
    print("=" * 60)
    
    # Preprocess data
    X_train, X_test, y_train, preprocessor, cat_features, num_features = preprocess_data(train_df, test_df)
    
    # Train model
    model = train_random_forest_model(X_train, y_train, preprocessor)
    
    # Generate predictions
    submission_df = generate_top3_predictions(model, X_test, test_df)
    
    print(f"\n Submission Summary:")
    print(f"  Submission shape: {submission_df.shape}")
    print(f"  Sample predictions:")
    print(submission_df.head(10))

if 'model' in locals():
    print("\nFeature Importance Analysis")
    print("=" * 60)
    
    try:
        # Get feature names after preprocessing
        feature_names = []
        
        # Add numerical features
        feature_names.extend(num_features)
        
        # Add one-hot encoded categorical features
        if cat_features:
            cat_feature_names = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_features)
            feature_names.extend(cat_feature_names)
        
        # Get feature importances
        importances = model.named_steps['classifier'].feature_importances_
        
        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"📈 Top 15 Most Important Features:")
        print(feature_importance_df.head(15).to_string(index=False))
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importances - Random Forest')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f" Error calculating feature importance: {e}")

if 'submission_df' in locals():
    print("\n Saving submission file")
    print("=" * 60)
    
    # Save to CSV
    submission_df.to_csv('submission.csv', index=False)
    
    print(f"Submission saved as 'submission.csv'")
    print(f"Final submission statistics:")
    print(f"Total predictions: {len(submission_df)}")
    print(f"File size: {os.path.getsize('submission.csv')} bytes")
    
    # Validate submission format
    print(f"\n Submission validation:")
    print(f" Required columns: {list(submission_df.columns) == ['id', 'Fertilizer Name']}")
    print(f" No missing values: {submission_df.isnull().sum().sum() == 0}")
    print(f" All IDs present: {len(submission_df['id'].unique()) == len(submission_df)}")
    
    # Sample submission content
    print(f"\n Sample submission content:")
    print(submission_df.head(10).to_string(index=False))
    
    # Download file
    print(f"\n Downloaded submission file...")
    files.download('submission.csv')

