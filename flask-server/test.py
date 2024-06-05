import pandas as pd
import re

def preprocess_text(text):
    # Remove extra spaces
    text = ' '.join(text.split())
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    
    return text

def preprocess_csv(input_file, output_file, column_name):
    # Read CSV file
    df = pd.read_csv(input_file)
    
    # Preprocess text in the specified column
    df[column_name] = df[column_name].apply(preprocess_text)
    
    # Save preprocessed data to a new CSV file
    df.to_csv(output_file, index=False)

# Example usage
input_file = '/Users/sandundesilva/Documents/4th year/Research Project/UI/findMyFilm/flask-server/Movie Final Filtered - movies3.csv'  # Provide the path to your input CSV file
output_file = 'output.csv'  # Provide the desired path for the output CSV file
column_name = 'Paragraph'  # Specify the column name containing the text to be preprocessed
preprocess_csv(input_file, output_file, column_name)
