import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import fitz
import os
import os

required_words = [
    "Sterilization",
    "Consent",
    "Permanent",
    "Irreversible",
    "Alternative",
    "Methods",
    "Birth",
    "Control",
    "Informed",
    "Decision",
    "30",
    "Day",
    "Waiting",
    "Period","Reversible","Mental","Competency","Withdrawal","Federal","Funding","Age","Interpreter","Risks","Benefits","Medical","Records","Physician","Witness","Emergency","Situations"
]


directory_path_for_yes = "yesConsent" 
file_names_yes = [os.path.join("yesConsent/",f)  for f in os.listdir(directory_path_for_yes) if os.path.isfile(os.path.join(directory_path_for_yes, f))]

directory_path_for_no = "noConsent"
file_names_no = [os.path.join("noConsent/",f) for f in os.listdir(directory_path_for_no) if os.path.isfile(os.path.join(directory_path_for_no, f))]

print(file_names_yes)
print(file_names_yes[0])


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def create_dataframe(file_names, label):
    data = []
    for file_name in file_names:
        text = extract_text_from_pdf(file_name)
        data.append({"text": text, "label": label})
    return pd.DataFrame(data)

def create_model_and_data(file_names_yes, file_names_no):
    df_yes = create_dataframe(file_names_yes, 1)
    df_no = create_dataframe(file_names_no, 0)
    df = pd.concat([df_yes, df_no])
    df.index = range(len(df))
    vectorizer = CountVectorizer(lowercase=True,vocabulary=required_words)
    vectorizer.fit(df["text"])
    X = vectorizer.transform(df["text"])
    y = df["label"] 
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    print("score:" ,model.score(X_test, y_test))
    print(model.predict(X_test))
    print(y_test)
    print(model.classes_)
    percentages = model.predict_proba(X_test) * 100

    for row in percentages:
        print(f"{row[0]:.2f}%  {row[1]:.2f}%")
    return score, percentages

model, X_train, X_test, y_train, y_test = create_model_and_data(file_names_yes, file_names_no)
score,percentages = evaluate_model(model, X_train, X_test, y_train, y_test)
