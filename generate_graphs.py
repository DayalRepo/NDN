import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_all_graphs():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'dataset', 'ndn_traffic.csv')
    assets_dir = os.path.join(base_dir, 'report_assets')

    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Please run dataset/generate_dummy_csv.py first.")
        return

    os.makedirs(assets_dir, exist_ok=True)

    df = pd.read_csv(dataset_path)

    # Group by 'Time' to get network-wide metrics (average or sum)
    # We will sum interests/load, and average ratios.
    time_grouped_sum = df.groupby('Time').sum().reset_index()
    time_grouped_mean = df.groupby('Time').mean(numeric_only=True).reset_index()

    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f8fafc", "grid.color": "#e2e8f0"})
    
    # Common function to save plots
    def save_plot(filename, title, xlabel, ylabel, data_x, data_y, color):
        plt.figure(figsize=(8, 4))
        sns.lineplot(x=data_x, y=data_y, color=color, linewidth=2)
        plt.fill_between(data_x, data_y, color=color, alpha=0.1)
        plt.title(title, fontsize=14, pad=10, fontname='sans-serif', fontweight='bold')
        plt.xlabel(xlabel, fontsize=11)
        plt.ylabel(ylabel, fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(assets_dir, filename), dpi=300)
        plt.close()

    # 1. Interest Rate Over Time
    save_plot('interest_rate.png', 'Interest Rate Over Time', 'Time (s)', 'Total Interest Rate', 
              time_grouped_sum['Time'], time_grouped_sum['interest_rate'], '#3182bd')

    # 2. Network Load (using total incoming data rate as proxy or interests+data)
    time_grouped_sum['network_load'] = time_grouped_sum['interest_rate'] + time_grouped_sum['data_rate']
    save_plot('network_load.png', 'Network Load Volatility', 'Time (s)', 'Total Network Load (Packets)', 
              time_grouped_sum['Time'], time_grouped_sum['network_load'], '#e6550d')

    # 3. PIT Occupancy
    save_plot('pit_occupancy.png', 'Pending Interest Table (PIT) Occupancy', 'Time (s)', 'Expected PIT Entries', 
              time_grouped_sum['Time'], time_grouped_sum['pit_estimate'], '#756bb1')

    # 4. Satisfaction Ratio
    save_plot('satisfaction_ratio.png', 'Network Satisfaction Ratio', 'Time (s)', 'Satisfaction %', 
              time_grouped_mean['Time'], time_grouped_mean['satisfaction_ratio'], '#31a354')

    # 5. Timeout Ratio
    save_plot('timeout_ratio.png', 'Packet Timeout Trend', 'Time (s)', 'Timeout Ratio', 
              time_grouped_mean['Time'], time_grouped_mean['timeout_ratio'], '#de2d26')

    # 6. NACK Ratio
    save_plot('nack_ratio.png', 'Network NACK Rate', 'Time (s)', 'NACK Ratio', 
              time_grouped_mean['Time'], time_grouped_mean['nack_ratio'], '#fdae6b')

    print("All graphs successfully generated in report_assets directory.")

if __name__ == "__main__":
    generate_all_graphs()
