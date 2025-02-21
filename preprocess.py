from datasets import load_dataset, Dataset
from huggingface_hub import login
import pandas as pd
import json
import argparse

def create_alpaca_prompt(title, description):
    """
    Creates an Alpaca-style prompt from title and description
    """
    # Create instruction from title
    instruction = f"{title}"
    
    # Use description as the response
    response = description
    
    # Combine into Alpaca format
    text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction: {instruction}\n\n### Response: {response}"
    
    return {
        "instruction": instruction,
        "output": response,
        "text": text
    }

def convert_to_alpaca_format(dataset):
    """
    Converts the dataset to Alpaca format
    """
    alpaca_data = {
        'instruction': [],
        'output': [],
        'text': []
    }
    
    for item in dataset:
        # Skip if title or description is missing
        if not item['title'] or not item['description']:
            continue
            
        # Create Alpaca format entry
        alpaca_entry = create_alpaca_prompt(
            title=item['title'],
            description=item['description']
        )
        
        # Add to respective lists
        alpaca_data['instruction'].append(alpaca_entry['instruction'])
        alpaca_data['output'].append(alpaca_entry['output'])
        alpaca_data['text'].append(alpaca_entry['text'])
    
    return alpaca_data

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert and upload dataset to Huggingface Hub')
    parser.add_argument('--token', type=str, default="", help='Huggingface token')
    parser.add_argument('--repo_name', type=str, default="mohit9999/all_news_finance_sm_1h2023_custom", help='Repository name for upload (e.g., username/dataset-name)')
    args = parser.parse_args()

    # Login to Huggingface Hub
    login(token=args.token)
    
    # Load the original dataset
    original_dataset = load_dataset("PaulAdversarial/all_news_finance_sm_1h2023")
    
    # Convert train split to Alpaca format
    train_data = convert_to_alpaca_format(original_dataset['train'])
    
    # Create Huggingface dataset
    hf_dataset = Dataset.from_dict(train_data)
    
    # Save locally
    hf_dataset.save_to_disk("all_news_finance_sm_1h2023_custom")
    
    # Push to Huggingface Hub
    hf_dataset.push_to_hub(
        args.repo_name,
        private=False,  # Set to True if you want a private repository
        commit_message="Upload financial news dataset in Alpaca format"
    )
    
    # Also save as CSV and JSON for backup
    df = pd.DataFrame(train_data)
    df.to_csv('alpaca_format_dataset.csv', index=False)
    
    with open('alpaca_format_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
        
    print(f"Converted {len(train_data['instruction'])} entries to Alpaca format")
    print(f"Dataset uploaded to Huggingface Hub: {args.repo_name}")
    print("Local backups saved as: alpaca_format_dataset.csv and alpaca_format_dataset.json")
    print(f"You can now use: dataset = load_dataset('{args.repo_name}')")

if __name__ == "__main__":
    main()