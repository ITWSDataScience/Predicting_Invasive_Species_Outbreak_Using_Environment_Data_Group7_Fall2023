import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

def map_impact_to_numeric(impact_text):
    if impact_text == '1 - Minimal Concern':
        return 1
    elif impact_text == '2 - Minor':
        return 2
    elif impact_text == '3 - Moderate':
        return 3
    elif impact_text == '4 - Major':
        return 4
    else:
        return 0

def filter_and_translate(data):
    print('Filtering data...')
    columns_to_keep = ['Growth form', 'Affected System', 'Affected Taxon', 'Mechanism', 'Impact']
    filtered_data = data[columns_to_keep].copy()  
    filtered_data = filtered_data[filtered_data['Impact'] != 'Not Available']
    filtered_data['Impact'] = filtered_data['Impact'].apply(map_impact_to_numeric)
    return filtered_data




if __name__ == "__main__":
    df = pd.read_excel('ny_species_impact.xlsx')
    print('DF loaded...')
    df = filter_and_translate(df)

    print('encoding...')
    label_encoder = LabelEncoder()
    df['Growth form'] = label_encoder.fit_transform(df['Growth form'])
    df['Affected System'] = label_encoder.fit_transform(df['Affected System'])
    df['Affected Taxon'] = label_encoder.fit_transform(df['Affected Taxon'])
    df['Mechanism'] = label_encoder.fit_transform(df['Mechanism'])

    # Split the data into training and testing sets
    X = df[['Growth form', 'Affected System', 'Affected Taxon', 'Mechanism']]
    y = df['Impact']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Support Vector Machine Classifier
    clf = SVC(kernel='linear', random_state=42)  # You can try different kernels like 'rbf', 'poly', etc.

    # Train the model
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))