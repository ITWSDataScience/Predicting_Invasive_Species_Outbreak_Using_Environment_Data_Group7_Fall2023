import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from itertools import product


def map_impact_to_numeric(impact_text):
    if impact_text == '1 - Minimal Concern':
        return 0
    elif impact_text == '2 - Minor':
        return 1
    elif impact_text == '3 - Moderate':
        return 2
    elif impact_text == '4 - Major':
        return 3
    else:
        return 4


def filter_and_translate(data):
    print('Filtering data...')
    columns_to_keep = ['Growth form', 'Affected System',
                       'Affected Taxon', 'Mechanism', 'Impact']
    filtered_data = data[columns_to_keep].copy()
    filtered_data = filtered_data[filtered_data['Impact'] != 'Not Available']
    filtered_data['Impact'] = filtered_data['Impact'].apply(
        map_impact_to_numeric)
    return filtered_data

def run_svm(X_train, y_train, X_test, y_test, kernel_type='linear'):
    # Initialize the Support Vector Machine Classifier
    clf = SVC(kernel=kernel_type, random_state=42)
    # Train the model
    clf.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Accuracy: {accuracy:.2f}")
    print("\nSVM Classification Report:")
    print(classification_report(y_test, y_pred))
    return clf

def run_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=None, random_state=42):
    # Initialize the Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    # Train the model
    clf.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy:.2f}")
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, y_pred))
    return clf

def run_xgb(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42):
    # Initialize the XGBoost Classifier
    clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=random_state)
    # Train the model
    clf.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Accuracy: {accuracy:.2f}")
    #print("\nXGBoost Classification Report:")
    #print(classification_report(y_test, y_pred))
    return clf

def hyper_xgb(X_train, y_train, X_test, y_test, n_estimators=100, learning_rates=[0.08, 0.1, 0.12], max_depths=[1, 2, 3]):
    best_accuracy = 0.0
    best_params = None

    for learning_rate, max_depth in product(learning_rates, max_depths):
        print(f"\nRunning XGBoost with Learning Rate: {learning_rate}, Max Depth: {max_depth}")
        # Call the run_xgb function with the specified hyperparameters
        clf = run_xgb(X_train, y_train, X_test, y_test, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
        # Make predictions on the test set
        y_pred = clf.predict(X_test)
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = (learning_rate, max_depth)

    print("\nBest Hyperparameters:")
    print(f"Learning Rate: {best_params[0]}, Max Depth: {best_params[1]}")
    print(f"Best Accuracy: {best_accuracy:.2f}")

    return best_params

def main():
    df = pd.read_excel('ny_species_impact.xlsx')
    print('Data size: ',df.size)
    print('DF loaded...')
    value_counts = df['Impact'].value_counts()
    print(value_counts)
    df = filter_and_translate(df)
    print('Data size: ',df.size)
    # Encode categorical columns
    label_encoder = LabelEncoder()
    df['Growth form'] = label_encoder.fit_transform(df['Growth form'])
    df['Affected System'] = label_encoder.fit_transform(df['Affected System'])
    df['Affected Taxon'] = label_encoder.fit_transform(df['Affected Taxon'])
    df['Mechanism'] = label_encoder.fit_transform(df['Mechanism'])

    # Split the data into training and testing sets
    X = df[['Growth form', 'Affected System', 'Affected Taxon', 'Mechanism']]
    y = df['Impact']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # SVM
    #print('Running SVM...')
    #svm_model = run_svm(X_train, y_train, X_test, y_test, kernel_type='linear')

    # Random Forest
    #print('\nRunning Random Forest...')
    #rf_model = run_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=None, random_state=42)

    # XGBoost
    print('\nRunning XGBoost...')
    xgb_model = run_xgb(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    best_hyperparameters = hyper_xgb(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
