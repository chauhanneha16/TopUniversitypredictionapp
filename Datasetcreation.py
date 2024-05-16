import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Create university data
data = {
    "University Name": [...],  # List of university names
    "University Link": [...],  # Corresponding links
    "Scholarship Info": [...],  # Scholarship links
    "Academic Fee": [...],  # Academic fees
    "Course": [...]  # Courses offered
}

df = pd.DataFrame(data)

# Create mock marks data
marks_data = {
    "Science Marks": np.random.randint(0, 101, size=len(df)),
    "Maths Marks": np.random.randint(0, 101, size=len(df)),
    "History Marks": np.random.randint(0, 101, size=len(df)),
    "English Marks": np.random.randint(0, 101, size=len(df)),
    "GRE Marks": np.random.randint(260, 341, size=len(df))
}
marks_df = pd.DataFrame(marks_data)

# Concatenate the marks data with the original dataframe
df = pd.concat([df, marks_df], axis=1)

# Encode the courses
label_encoder_course = LabelEncoder()
df["Course"] = label_encoder_course.fit_transform(df["Course"])

# Save the dataset
df.to_csv("university_data.csv", index=False)

# Save the course label encoder
with open('label_encoder_course.pkl', 'wb') as f:
    pickle.dump(label_encoder_course, f)
