import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import fitz
import os

# Required words list
required_words = [
     "Consent", "Permanent", "Alternative",
    "Methods", "Birth", "Control", "Informed", "Decision", "30", "Day",
    "Waiting", "Period", "Reversible", "Mental", "Competency", "Withdrawal",
    "Federal", "Funding", "Age", "Interpreter", "Risks", "Benefits", "Medical",
    "Records", "Physician", "Witness", "Emergency", "Situations"
]

# Assign custom weights to key terms (higher for more critical terms)

word_weights = {
    "Consent": 2.5,
    "Interpreter": 4.0,
    "Informed": 1.8,
    "Alternative": 1.5,
    "Decision": 1.5,
    "Permanent": 2.0,
    "Withdrawal": 1.8,
    "Competency": 1.8,
    "Emergency": 2.2,
    "Federal": 1.5,
    "Witness": 1.8,
    "Physician": 1.6,
    "Risks": 1.4,
    "Benefits": 1.4,
    '30':2.0
}

# File directories
namelist = []
directory_path_for_yes = "yesConsent"
file_names_yes = [os.path.join(directory_path_for_yes, f) for f in os.listdir(directory_path_for_yes) if os.path.isfile(os.path.join(directory_path_for_yes, f))]

directory_path_for_no = "noConsent"
file_names_no = [os.path.join(directory_path_for_no, f) for f in os.listdir(directory_path_for_no) if os.path.isfile(os.path.join(directory_path_for_no, f))]
namelist = []+file_names_yes+file_names_no


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# Function to create dataframe
def create_dataframe(file_names, label):
    data = []
    for file_name in file_names:
        text = extract_text_from_pdf(file_name)
        data.append({"text": text, "label": label})
    return pd.DataFrame(data)


# Function to apply custom weights
def apply_custom_weights(X, vectorizer):
    X = X.toarray()
    for word, weight in word_weights.items():
        if word.lower() in vectorizer.vocabulary_:
            index = vectorizer.vocabulary_[word.lower()]
            X[:, index] *= weight
    return X


# Function to create model and data
def create_model_and_data(file_names_yes, file_names_no):
    df_yes = create_dataframe(file_names_yes, 1)
    df_no = create_dataframe(file_names_no, 0)
    df = pd.concat([df_yes, df_no])
    df.index = range(len(df))

    vectorizer = TfidfVectorizer() # lowercase=True, vocabulary=[w.lower() for w in required_words]
    X = vectorizer.fit_transform(df["text"])

    # Apply custom word weighting
    X = apply_custom_weights(X, vectorizer)

    y = df["label"]
    model = LogisticRegression()

    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, train_indices, test_indices = sk.model_selection.train_test_split(
    X, y, indices, test_size=0.2, random_state=42)

    print("The test indices are:", test_indices)

    print([namelist[x] for x in test_indices])
    # No class weights in the model
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test,vectorizer


# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)

    print("Model Score:", score)
    print("Predictions:", y_pred)
    print("Actual:", y_test.values)
    print("Classes:", model.classes_)

    percentages = model.predict_proba(X_test) * 100
    for row in percentages:
        print(f"{row[0]:.2f}%  {row[1]:.2f}%")

    return score, percentages

def predict_example(model, vectorizer, new_file_name):
    text = extract_text_from_pdf(new_file_name)

    X = vectorizer.transform([text])

    X = apply_custom_weights(X, vectorizer)

    prediction = model.predict(X)[0]  
    probability = model.predict_proba(X)[0]  
    print(f"Prediction: {prediction}")
    print(f"Probability of class 0: {probability[0]:.2f}")
    print(f"Probability of class 1: {probability[1]:.2f}")

    return prediction, probability


# Run the pipeline
model, X_train, X_test, y_train, y_test,vectorizer = create_model_and_data(file_names_yes, file_names_no)
score, percentages = evaluate_model(model, X_test, y_test)

print(predict_example(model, vectorizer, "/Users/terrelldavis/Desktop/Personal Projects/H2aiV2hackathon/all files/CFR-2007-title42-vol1-part50-subpartB-app-id376 copy.pdf"))
# change to accept input later
