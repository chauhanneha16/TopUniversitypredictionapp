import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
df = pd.read_csv("university_data.csv")

# Preprocess the data
marks_columns = ["Science Marks", "Maths Marks", "History Marks", "English Marks", "GRE Marks"]
scaler = StandardScaler()
df[marks_columns] = scaler.fit_transform(df[marks_columns])

X = df[marks_columns]
y = df["University Name"]

# Encode the university names
label_encoder_uni = LabelEncoder()
y_encoded = label_encoder_uni.fit_transform(y)

# Save the scaler and label encoder
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder_uni.pkl', 'wb') as f:
    pickle.dump(label_encoder_uni, f)

# Build the model
model = Sequential([
    Dense(64, input_dim=X.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y_encoded, epochs=50, batch_size=4)

# Save the model
model.save('university_recommendation_model.h5')
