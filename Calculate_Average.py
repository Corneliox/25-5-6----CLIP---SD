import pandas as pd

# Load the CSV file into a DataFrame
file_path = 'combined_CLIP_scores.csv'  # Update this path if needed
df = pd.read_csv(file_path)

# Check if required columns exist
if 'Prompt_Variant' in df.columns and 'Score' in df.columns:
    # Group by version and select top 5 scores for each
    top_5_scores = df.groupby('Prompt_Variant').apply(
        lambda x: x.sort_values(by='Score', ascending=False).head(5)
    ).reset_index(drop=True)

    # Calculate the average of the top 5 scores per version
    average_scores = top_5_scores.groupby('Prompt_Variant')['Score'].mean().reset_index()
    average_scores.columns = ['Version', 'Average_Top_5_Score']

    # Sort the results by version (optional)
    average_scores = average_scores.sort_values(by='Version')

    # Display the result
    print("\nTop 5 Average Scores per Version:")
    print(average_scores)

    # Save the result to a new CSV file
    output_file = 'average_top_5_scores.csv'
    average_scores.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
else:
    print("Error: Required columns 'Prompt_Variant' or 'Score' not found in the CSV.")