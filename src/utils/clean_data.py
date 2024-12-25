import argparse
import re

def remove_emojis_and_non_ascii(text):
    # Remove emojis and non-ASCII characters
    return re.sub(r'[^\x00-\x7F]+', '', text)

# Preprocessing function
def preprocess_tweets(raw_data):
    # Combine tweets into a single string
    combined_data = " ".join(raw_data)
    
    # Remove URLs
    combined_data = re.sub(r'http\S+|www\S+|https\S+', '', combined_data, flags=re.MULTILINE)
    
    # Remove special tokens like @mentions and #hashtags
    combined_data = re.sub(r'@\w+|#\w+', '', combined_data)
    
    # Remove non-printable characters and extra whitespace
    combined_data = re.sub(r'[^ -~]+', '', combined_data)  # Keep printable ASCII characters
    combined_data = re.sub(r'\s+', ' ', combined_data).strip()  # Normalize whitespace
    
    return combined_data

def main():
    parser = argparse.ArgumentParser(description='Remove emojis and non-ASCII symbols from a text file.')
    parser.add_argument('input_file', type=str, help='Path to the input text file')
    parser.add_argument('output_file', type=str, help='Path to the output text file')
    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as file:
        text = file.readlines()

    cleaned_text = remove_emojis_and_non_ascii(text)
    cleaned_text = preprocess_tweets(cleaned_text)

    # Save the cleaned dataset
    output_path = 'data_elon_cleaned.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

    print(f"Cleaned dataset saved to: {output_path}")

if __name__ == '__main__':
    main()