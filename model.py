import pandas as pd

dataset = pd.read_excel('PublicLEEDProjectsDirectory.xlsx', engine="openpyxl")

# Apply the data filtering logic
df = dataset[~((dataset['IsCertified'] == 'No') | (dataset['Country'] != 'US'))]
df = df.drop('TotalPropArea', axis=1)
df = df.dropna(subset=['ProjectTypes'])
df['OwnerTypes'].fillna(value='Others', inplace=True)

# replace OwnerTypes
df['OwnerTypes'] = df['OwnerTypes'].replace(to_replace={
    r'(?i).*Investor.*': 'Investor',
    r'(?i).*Educational.*': 'Educational',
    r'(?i).*Government.*': 'Government Use',
    r'(?i).*Community.*': 'Community Development Corporation',
    r'(?i).*Corporate.*': 'Corporate',
    r'(?i).*(Profit|Confidential|Other|Individual|Business Improvement District|Main Street Organization).*': 'Others'
}, regex=True)

# replace ProjectTypes
df['ProjectTypes'] = df['ProjectTypes'].replace(to_replace={
    r'(?i).*Office.*': 'Office',
    r'(?i).*Retail.*': 'Retail',
    r'(?i).*Core.*': 'Core learning',
    r'(?i).*Assembly.*': 'Public Assembly',
    r'(?i).*(Lodging|Home).*': 'Lodging',
    r'(?i).*Restaurant.*': 'Service',
    r'(?i).*Service.*': 'Service',
    r'(?i).*Recreation.*': 'Service',
    r'(?i).*(Care|Daycare|Healthcare).*': 'Health Care',
    r'(?i).*Warehouse.*': 'Warehouse and distribution centre',
    r'(?i).*Industrial.*': 'Industrial Manufacturing',
    r'(?i).*Multi-Unit Residence.*': 'Multi-Unit Residence',
    r'(?i).*Public.*': 'Public Order and Safety',
    r'(?i).*(Residential|family|MF).*': 'Multi-Unit Residence',
    r'(?i).*Education.*': 'Education',
    r'(?i).*(School|Campus).*': 'Education',
    r'(?i).*Military.*': 'Military Base',
    r'(?i).*(Airport|ND|Other|Laboratory|Community|Interpretive|Confidential|Transportation|Financial|Special|Datacenter|Park|Transit|Library|HOtel|stadium).*': 'Others'
}, regex=True)


selected_columns = ['GrossFloorArea', 'State','OwnerTypes','ProjectTypes','PointsAchieved','CertLevel']
df = df[selected_columns]

# Drop rows with NaN values
df = df.dropna()

# Reset the index
df = df.reset_index(drop=True)


# Replace 0 with 'Others' in 'ProjectTypes' column
df['ProjectTypes'] = df['ProjectTypes'].replace('0', 'Others')
df['ProjectTypes'] = df['ProjectTypes'].replace('USA', 'Others')



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Convert float columns to string
df['State'] = df['State'].astype(str)
df['OwnerTypes'] = df['OwnerTypes'].astype(str)
df['ProjectTypes'] = df['ProjectTypes'].astype(str)


# Extract features and target variable
X = df[['GrossFloorArea', 'State', 'OwnerTypes', 'ProjectTypes']]
y = df['CertLevel']

# Perform label encoding for categorical variables
label_encoder_state = LabelEncoder()
X['State'] = label_encoder_state.fit_transform(X['State'])

label_encoder_owner_types = LabelEncoder()
X['OwnerTypes'] = label_encoder_owner_types.fit_transform(X['OwnerTypes'])

label_encoder_project_types = LabelEncoder()
X['ProjectTypes'] = label_encoder_project_types.fit_transform(X['ProjectTypes'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Initialize the models
knn = KNeighborsClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

 #Train the models
knn.fit(X_train, y_train)
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)

# Make predictions on the test data
y_pred_knn = knn.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_lr = lr.predict(X_test)

# Evaluate the models using accuracy score
accuracy_knn = accuracy_score(y_test, y_pred_knn)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

# Print the accuracy of each model
print('KNN Accuracy:', accuracy_knn)
print('Random Forest Accuracy:', accuracy_rf)
print('Logistic Regression Accuracy:', accuracy_lr)




import pickle

# Save the RF model
with open('rf_model.pkl', 'wb') as file:
    pickle.dump(rf, file)

    # Save the trained LabelEncoder objects
with open('label_encoder_state.pkl', 'wb') as file:
    pickle.dump(label_encoder_state, file)

with open('label_encoder_owner_types.pkl', 'wb') as file:
    pickle.dump(label_encoder_owner_types, file)

with open('label_encoder_project_types.pkl', 'wb') as file:
    pickle.dump(label_encoder_project_types, file)

print("Encoded classes for 'State':", label_encoder_state.classes_)
print("Encoded classes for 'OwnerTypes':", label_encoder_owner_types.classes_)
print("Encoded classes for 'ProjectTypes':", label_encoder_project_types.classes_)