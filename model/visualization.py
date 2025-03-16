#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

#%%
def visualize_synthetic_data(csv_path='/Users/sayam_palrecha/my_project/Hackathon/train_data.csv'):
    """
    Create visualizations for the synthetic TennCare sterilization consent form data

    Parameters:
    csv_path (str): Path to the CSV file containing the synthetic data
    """
    # Load the data
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded data from {csv_path}")
        print(f"Dataset contains {len(df)} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Create output directory for visualizations
    output_dir = 'synthetic_data_visualizations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set plot style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # 1. Distribution of Operation Types
    plt.figure(figsize=(12, 6))
    op_counts = df['SpecifyTypeOfOperation'].value_counts()
    ax = sns.barplot(x=op_counts.index, y=op_counts.values)
    plt.title('Distribution of Operation Types', fontsize=16)
    plt.xlabel('Operation Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')

    # Add count labels on top of bars
    for i, count in enumerate(op_counts.values):
        ax.text(i, count + 50, f'{count:,}', ha='center')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/operation_type_distribution.png")
    plt.close()

    # 2. Distribution of Race/Ethnicity
    plt.figure(figsize=(10, 6))
    race_counts = df['RaceAndEthnicDesignation'].value_counts()
    ax = sns.barplot(x=race_counts.index, y=race_counts.values)
    plt.title('Distribution of Race/Ethnicity', fontsize=16)
    plt.xlabel('Race/Ethnicity')
    plt.ylabel('Count')

    # Add count labels on top of bars
    for i, count in enumerate(race_counts.values):
        ax.text(i, count + 50, f'{count:,}', ha='center')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/race_ethnicity_distribution.png")
    plt.close()

    # 3. Distribution of Waiting Period Days
    plt.figure(figsize=(14, 6))
    plt.hist(df['WaitingPeriodDays'], bins=50, color='skyblue', edgecolor='black')
    plt.axvline(x=30, color='red', linestyle='--', linewidth=2, label='30-day threshold')
    plt.title('Distribution of Waiting Period Days', fontsize=16)
    plt.xlabel('Days')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/waiting_period_distribution.png")
    plt.close()

    # 4. Compliance by Race/Ethnicity
    plt.figure(figsize=(10, 6))
    compliance_by_race = df.groupby('RaceAndEthnicDesignation')['WaitingPeriodFlag'].mean() * 100
    compliance_by_race = compliance_by_race.sort_values(ascending=False)

    ax = sns.barplot(x=compliance_by_race.index, y=compliance_by_race.values, palette='coolwarm')
    plt.title('Non-Compliance Rate by Race/Ethnicity', fontsize=16)
    plt.xlabel('Race/Ethnicity')
    plt.ylabel('Non-Compliance Rate (%)')

    # Add percentage labels on top of bars
    for i, rate in enumerate(compliance_by_race.values):
        ax.text(i, rate + 0.5, f'{rate:.1f}%', ha='center')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/compliance_by_race.png")
    plt.close()

    # 5. Compliance by Operation Type
    plt.figure(figsize=(12, 6))
    compliance_by_op = df.groupby('SpecifyTypeOfOperation')['WaitingPeriodFlag'].mean() * 100
    compliance_by_op = compliance_by_op.sort_values(ascending=False)

    ax = sns.barplot(x=compliance_by_op.index, y=compliance_by_op.values, palette='coolwarm')
    plt.title('Non-Compliance Rate by Operation Type', fontsize=16)
    plt.xlabel('Operation Type')
    plt.ylabel('Non-Compliance Rate (%)')
    plt.xticks(rotation=45, ha='right')

    # Add percentage labels on top of bars
    for i, rate in enumerate(compliance_by_op.values):
        ax.text(i, rate + 0.5, f'{rate:.1f}%', ha='center')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/compliance_by_operation.png")
    plt.close()

    # 6. Interpreter Usage by Race/Ethnicity
    plt.figure(figsize=(10, 6))
    interpreter_by_race = df.groupby('RaceAndEthnicDesignation')['InterpreterSignature'].mean() * 100
    interpreter_by_race = interpreter_by_race.sort_values(ascending=False)

    ax = sns.barplot(x=interpreter_by_race.index, y=interpreter_by_race.values, palette='viridis')
    plt.title('Interpreter Usage Rate by Race/Ethnicity', fontsize=16)
    plt.xlabel('Race/Ethnicity')
    plt.ylabel('Interpreter Usage Rate (%)')

    # Add percentage labels on top of bars
    for i, rate in enumerate(interpreter_by_race.values):
        ax.text(i, rate + 0.5, f'{rate:.1f}%', ha='center')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/interpreter_by_race.png")
    plt.close()

    # 7. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    # Create binary columns for categorical variables
    corr_df = pd.get_dummies(df[['RecipientSignature', 'InterpreterSignature', 'WaitingPeriodFlag',
                                 'RaceAndEthnicDesignation', 'SpecifyTypeOfOperation']])

    # Calculate correlation matrix
    corr = corr_df.corr()

    # Plot heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, annot=False, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()

    # 8. Signature Status Distribution
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    recipient_counts = df['RecipientSignature'].value_counts()
    labels = ['Signed', 'Not Signed']
    plt.pie(recipient_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
    plt.title('Recipient Signature Status')

    plt.subplot(1, 2, 2)
    interpreter_counts = df['InterpreterSignature'].value_counts()
    labels = ['Not Used', 'Used']
    plt.pie(interpreter_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#99ff99', '#ffcc99'])
    plt.title('Interpreter Signature Status')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/signature_distribution.png")
    plt.close()

    # 9. Waiting Period Compliance Summary
    plt.figure(figsize=(8, 6))
    compliance_counts = df['WaitingPeriodFlag'].value_counts()
    labels = ['Compliant (>30 days)', 'Non-Compliant (<30 days)']
    explode = (0, 0.1)  # Explode the second slice (non-compliant)

    plt.pie(compliance_counts, explode=explode, labels=labels,
            autopct='%1.1f%%', startangle=90, shadow=True,
            colors=['#66b3ff', '#ff9999'])

    plt.title('Waiting Period Compliance', fontsize=16)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.tight_layout()
    plt.savefig(f"{output_dir}/compliance_summary.png")
    plt.close()

    # 10. Cross-tabulation: Operation Type vs Compliance
    plt.figure(figsize=(12, 8))

    # Create a crosstab
    ct = pd.crosstab(df['SpecifyTypeOfOperation'], df['WaitingPeriodFlag'],
                     normalize='index') * 100
    ct.columns = ['Compliant', 'Non-Compliant']

    # Plot stacked bar chart
    ct.plot(kind='barh', stacked=True, color=['#66b3ff', '#ff9999'], figsize=(12, 8))
    plt.title('Compliance Rate by Operation Type', fontsize=16)
    plt.xlabel('Percentage')
    plt.ylabel('Operation Type')
    plt.legend(title='Compliance Status')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/operation_compliance_crosstab.png")
    plt.close()

    print(f"Visualizations saved to the '{output_dir}' directory")


if __name__ == "__main__":
    import sys

    # Filter out web server arguments
    args = [arg for arg in sys.argv if not arg.startswith('--')]

    if len(args) > 1:
        visualize_synthetic_data(args[1])
    else:
        visualize_synthetic_data()