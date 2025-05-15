import pandas as pd

# Load the CSV file into a DataFrame
file_path = 'combined_CLIP_scores.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Display the first few rows to check the structure
print("First few rows of the data:")
print(df.head())

# Check if 'Prompt_Variant1' and 'Score' columns exist
if 'Prompt_Variant' in df.columns and 'Score' in df.columns:
    # Group by 'Prompt_Variant1' and calculate average score
    average_scores = df.groupby('Prompt_Variant')['Score'].mean().reset_index()
    average_scores.columns = ['Version', 'Average_Score']

    # Sort by version for better readability (optional)
    average_scores = average_scores.sort_values(by='Version')

    # Display the result
    print("\nAverage CLIP Scores per Version:")
    print(average_scores)

    # Save the result to a new CSV file
    output_file = 'average_scores_per_version.csv'
    average_scores.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
else:
    print("Error: Required columns 'Prompt_Variant' or 'Score' not found in the CSV.")