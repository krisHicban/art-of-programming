# ========================================
# PART 1: Import Libraries
# ========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

# For text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# For machine learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

# Download NLTK data (run once)
# nltk.download('stopwords')
# nltk.download('punkt')

# Set up visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ========================================
# PART 2: Load and Explore Data
# ========================================
print("Loading dataset...")
# Load the dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only the relevant columns
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Convert labels to binary (0 for ham, 1 for spam)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Basic exploration
print(f"Dataset shape: {df.shape}")
print(f"\nLabel distribution:\n{df['label'].value_counts()}")
print(f"\nPercentage of spam: {df['label'].mean()*100:.2f}%")

# Add text statistics
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

print(f"\nAverage text length - Ham: {df[df['label']==0]['text_length'].mean():.2f}")
print(f"Average text length - Spam: {df[df['label']==1]['text_length'].mean():.2f}")

# Visualize label distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot of label distribution
df['label'].value_counts().plot(kind='bar', ax=axes[0], color=['green','red'])
axes[0].set_title("Distribution of Messages", fontsize=14, fontweight='bold')
axes[0].set_xticklabels(['Ham (Legitimate)', 'Spam'], rotation=0)
axes[0].set_xlabel('Message Type')
axes[0].set_ylabel('Count')

# Box plot of text length by label
df.boxplot(column='text_length', by='label', ax=axes[1])
axes[1].set_title("Text Length Distribution", fontsize=14, fontweight='bold')
axes[1].set_xlabel('Message Type (0=Ham, 1=Spam)')
axes[1].set_ylabel('Character Count')
plt.suptitle('')  # Remove default title
plt.tight_layout()
plt.show()

# Show sample messages
print("\n" + "="*50)
print("SAMPLE HAM MESSAGES:")
print("="*50)
for i, text in enumerate(df[df['label']==0]['text'].head(3), 1):
    print(f"{i}. {text[:100]}...")
    
print("\n" + "="*50)
print("SAMPLE SPAM MESSAGES:")
print("="*50)
for i, text in enumerate(df[df['label']==1]['text'].head(3), 1):
    print(f"{i}. {text[:100]}...")

# ========================================
# PART 3: Text Preprocessing
# ========================================
print("\n" + "="*50)
print("PREPROCESSING TEXT...")
print("="*50)

# Initialize preprocessing tools
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess_text(text):
    """
    Clean and preprocess text data:
    1. Convert to lowercase
    2. Remove numbers
    3. Remove punctuation
    4. Remove stopwords
    5. Apply stemming
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Apply stemming
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)

# Show preprocessing results
print("BEFORE PREPROCESSING:")
print(df['text'].iloc[0][:150])
print("\nAFTER PREPROCESSING:")
print(df['clean_text'].iloc[0][:150])

# ========================================
# PART 4: Feature Extraction (TF-IDF)
# ========================================
print("\n" + "="*50)
print("VECTORIZING TEXT WITH TF-IDF...")
print("="*50)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=3000,  # Use top 3000 most important words
    min_df=2,           # Ignore words that appear in less than 2 documents
    max_df=0.95,        # Ignore words that appear in more than 95% of documents
    ngram_range=(1, 2)  # Use both unigrams and bigrams
)

# Transform text to numerical features
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

print(f"Feature matrix shape: {X.shape}")
print(f"Number of features (words/bigrams): {X.shape[1]}")

# Get most important features for spam
feature_names = vectorizer.get_feature_names_out()
tfidf_sum = X.toarray().sum(axis=0)
top_idx = tfidf_sum.argsort()[-20:][::-1]
top_features = [(feature_names[i], tfidf_sum[i]) for i in top_idx]

print("\nTop 10 most important features overall:")
for i, (feature, score) in enumerate(top_features[:10], 1):
    print(f"{i:2d}. {feature:20s} (score: {score:.2f})")

# ========================================
# PART 5: Split Data
# ========================================
print("\n" + "="*50)
print("SPLITTING DATA...")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Training spam ratio: {y_train.mean()*100:.2f}%")
print(f"Test spam ratio: {y_test.mean()*100:.2f}%")

# ========================================
# PART 6: Train Multiple Models
# ========================================
print("\n" + "="*50)
print("TRAINING MODELS...")
print("="*50)

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC(kernel='linear', probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    print(f"{name} Accuracy: {accuracy:.4f}")

# ========================================
# PART 7: Model Evaluation
# ========================================
print("\n" + "="*50)
print("DETAILED EVALUATION...")
print("="*50)

# Find best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
print(f"\nBest Model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")

# Print detailed classification report for best model
print(f"\n{best_model_name} Classification Report:")
print("="*50)
print(classification_report(y_test, results[best_model_name]['y_pred'], 
                          target_names=['Ham', 'Spam']))

# Create visualization subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Confusion Matrix for best model
cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xticklabels(['Ham', 'Spam'])
axes[0, 0].set_yticklabels(['Ham', 'Spam'])

# 2. Model Comparison
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
colors = ['green' if acc == max(accuracies) else 'skyblue' for acc in accuracies]

axes[0, 1].bar(model_names, accuracies, color=colors)
axes[0, 1].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_ylim([0.9, 1.0])
axes[0, 1].tick_params(axis='x', rotation=45)
for i, v in enumerate(accuracies):
    axes[0, 1].text(i, v + 0.001, f'{v:.4f}', ha='center')

# 3. Precision, Recall, F1-Score Comparison
metrics = ['precision', 'recall', 'f1-score']
x = np.arange(len(model_names))
width = 0.25

for i, metric in enumerate(metrics):
    values = [results[name]['report']['1'][metric] for name in model_names]
    axes[1, 0].bar(x + i*width, values, width, label=metric.capitalize())

axes[1, 0].set_title('Spam Detection Metrics Comparison', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_xlabel('Models')
axes[1, 0].set_xticks(x + width)
axes[1, 0].set_xticklabels(model_names, rotation=45)
axes[1, 0].legend()
axes[1, 0].set_ylim([0.8, 1.0])

# 4. ROC Curves
for name in results:
    if results[name]['y_pred_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test, results[name]['y_pred_proba'])
        auc = roc_auc_score(y_test, results[name]['y_pred_proba'])
        axes[1, 1].plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random')
axes[1, 1].set_title('ROC Curves', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].legend(loc='lower right')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========================================
# PART 8: Test with New Messages
# ========================================
print("\n" + "="*50)
print("TESTING WITH NEW MESSAGES...")
print("="*50)

# Create a function to predict new messages
def predict_spam(text, model_name='Logistic Regression'):
    """Predict if a message is spam or not"""
    # Preprocess the text
    clean = preprocess_text(text)
    
    # Vectorize
    text_vector = vectorizer.transform([clean])
    
    # Get model
    model = results[model_name]['model']
    
    # Predict
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    
    return {
        'text': text,
        'is_spam': bool(prediction),
        'spam_probability': probability[1],
        'ham_probability': probability[0]
    }

# Test messages
test_messages = [
    "Hey, are we still meeting for lunch tomorrow?",
    "WINNER! You've won £1000! Click here to claim your prize NOW!",
    "Your package will be delivered tomorrow between 2-4 PM",
    "FREE VIAGRA!!! Buy now and get 50% OFF! Limited time offer!!!",
    "Can you pick up some milk on your way home?",
    "Congratulations! You've been selected for a free iPhone. Reply YES to claim!"
]

print(f"\nUsing {best_model_name} for predictions:\n")
for msg in test_messages:
    result = predict_spam(msg, best_model_name)
    label = "🚫 SPAM" if result['is_spam'] else "✅ HAM"
    confidence = result['spam_probability'] if result['is_spam'] else result['ham_probability']
    print(f"{label} (Confidence: {confidence:.2%})")
    print(f"Message: {msg[:80]}{'...' if len(msg) > 80 else ''}")
    print("-" * 70)

# ========================================
# PART 9: Save the Model
# ========================================
print("\n" + "="*50)
print("SAVING MODEL...")
print("="*50)

import joblib

# Save the best model and vectorizer
best_model = results[best_model_name]['model']
joblib.dump(best_model, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print(f"✅ Model saved as 'spam_classifier_model.pkl'")
print(f"✅ Vectorizer saved as 'tfidf_vectorizer.pkl'")

# ========================================
# PART 10: Summary Statistics
# ========================================
print("\n" + "="*50)
print("PROJECT SUMMARY")
print("="*50)
print(f"📊 Total messages processed: {len(df)}")
print(f"📈 Features extracted: {X.shape[1]}")
print(f"🎯 Best model: {best_model_name}")
print(f"✨ Best accuracy: {results[best_model_name]['accuracy']:.2%}")
print(f"🚫 Spam detection rate (Recall): {results[best_model_name]['report']['1']['recall']:.2%}")
print(f"✅ Spam precision: {results[best_model_name]['report']['1']['precision']:.2%}")