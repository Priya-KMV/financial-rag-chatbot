import pandas as pd

# Load the structured data
df = pd.read_csv("data/Financial_Data.csv")

# Clean any commas/spaces and convert the numeric columns to real numbers
num_cols = ['Total Revenue', 'Net Income', 'Total Assets',
            'Total Liabilities', 'Operating Cash Flow']
for c in num_cols:
    df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '').str.strip(), errors='coerce')

def row_to_text(row):
    """Convert one row of financial data into a natural-language sentence."""
    return (
        f"In fiscal year {int(row['Fiscal Year'])}, {row['Company']} reported "
        f"total revenue of ${row['Total Revenue']/1000:,.1f} billion, "
        f"net income of ${row['Net Income']/1000:,.1f} billion, "
        f"total assets of ${row['Total Assets']/1000:,.1f} billion, "
        f"total liabilities of ${row['Total Liabilities']/1000:,.1f} billion, "
        f"and operating cash flow of ${row['Operating Cash Flow']/1000:,.1f} billion."
    )

# Build one text document per row
documents = [row_to_text(row) for _, row in df.iterrows()]

# Show what we created
for i, doc in enumerate(documents):
    print(f"[Document {i}] {doc}\n")

print(f"Total documents created: {len(documents)}")