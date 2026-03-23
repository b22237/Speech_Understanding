import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os

# --- Configuration ---
# Pointing specifically to your Spontaneous Speech TSV file
FILE_PATH = '/home/preet-savalia/Downloads/speech_assi1/q3/sps-corpus-3.0-2026-03-09-en/ss-corpus-en.tsv' 
OUTPUT_PDF = 'audit_plots.pdf'

# Updated to match your exact schema ('accents' plural)
TARGET_COLS = ['gender', 'age', 'accents']

def perform_spontaneous_speech_audit():
    if not os.path.exists(FILE_PATH):
        print(f"Error: Could not find '{FILE_PATH}'.")
        print("Ensure the dataset file is extracted and in the same directory as this script.")
        return

    print(f"Loading '{FILE_PATH}'...")
    # TSV means Tab-Separated Values
    df = pd.read_csv(FILE_PATH, sep='\t', low_memory=False)

    print("\n" + "="*45)
    print("      ETHICAL AUDIT: DOCUMENTATION DEBT")
    print("="*45)
    
    total_samples = len(df)
    print(f"Total audio clips analyzed: {total_samples:,}\n")

    for col in TARGET_COLS:
        if col in df.columns:
            # Check for NaN or empty space strings
            missing_count = df[col].isna().sum() + (df[col] == ' ').sum() + (df[col] == '').sum()
            debt_pct = (missing_count / total_samples) * 100
            print(f"Feature: {col:<8} | Missing: {missing_count:>6,} | Debt: {debt_pct:>6.2f}%")
        else:
            print(f"Feature: {col:<8} | ERROR: Column not found in TSV schema.")

    print("="*45)

    print(f"\nGenerating Representation Bias plots -> {OUTPUT_PDF}")
    sns.set_theme(style="whitegrid")
    
    with PdfPages(OUTPUT_PDF) as pdf:
        for col in TARGET_COLS:
            if col in df.columns:
                # Clean the data: drop NaNs and empty strings for the chart
                clean_df = df.dropna(subset=[col])
                clean_df = clean_df[clean_df[col].str.strip() != ''] 
                
                if not clean_df.empty:
                    plt.figure(figsize=(10, 6))
                    
                    # Sort by frequency so the largest demographic is on the left
                    order = clean_df[col].value_counts().index
                    
                    ax = sns.countplot(data=clean_df, x=col, order=order, palette='mako', hue=col, legend=False)
                    
                    plt.title(f'Representation Bias: {col.capitalize()} Distribution', fontsize=14, pad=15)
                    plt.xlabel(col.capitalize(), fontsize=12)
                    plt.ylabel('Number of Audio Samples', fontsize=12)
                    plt.xticks(rotation=45, ha='right')
                    
                    # Add count numbers on top of the bars
                    for p in ax.patches:
                        ax.annotate(f'{int(p.get_height()):,}', 
                                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                                    ha='center', va='center', 
                                    xytext=(0, 9), 
                                    textcoords='offset points')
                    
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
                    print(f" -> Plotted {col} distribution.")
                else:
                    print(f" -> Skipping {col}: No labeled data to plot.")

    print(f"\nAudit Complete! Check your folder for '{OUTPUT_PDF}'.")

if __name__ == "__main__":
    perform_spontaneous_speech_audit()