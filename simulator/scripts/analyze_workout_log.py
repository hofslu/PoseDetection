import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


"""
This script analyzes a workout log stored in a JSON file.
It reads the data, processes it, and generates various visualizations to help understand workout patterns and performance over time.
Clusters the data by rest break separations (longer than 1 minute) and generates a heatmap of the workout data.
Sample Data:
[
  {
    "timestamp": "2025-04-08T00:45:03.160016",
    "similarity_up": 91.0,
    "similarity_down": 73.2
  },
  {
    "timestamp": "2025-04-08T00:45:06.900484",
    "similarity_up": 88.75,
    "similarity_down": 74.88
  },
  ...
]
"""


def load_workout_log(file_path):
    """
    Load the workout log from a JSON file.
    :param file_path: Path to the JSON file.
    :return: DataFrame containing the workout log.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def process_workout_log(df):
    """
    Process the workout log DataFrame to extract relevant information.
    :param df: DataFrame containing the workout log.
    :return: Processed DataFrame.
    """
    # Rename timestamp column to date and convert to datetime format
    df['date'] = pd.to_datetime(df['timestamp'])
    
    # Calculate time difference between consecutive entries (in minutes)
    df['time_diff'] = df['date'].diff().dt.total_seconds() / 60  # in minutes
    print(df['time_diff'])
    
    # Identify rest breaks longer than 1 minute
    df['rest_break'] = df['time_diff'] > 1  # True if rest break > 1 minute
    print(df['rest_break'])
    
    # Cluster based on breaks, start a new cluster when rest_break is True
    df['cluster'] = df['rest_break'].cumsum()
    
    # Count pushups within each cluster
    df['count'] = df.groupby('cluster').cumcount() + 1
    
    # Calculate the average similarity for each cluster
    df['avg_similarity_up'] = df.groupby('cluster')['similarity_up'].transform('mean')
    df['avg_similarity_down'] = df.groupby('cluster')['similarity_down'].transform('mean')
    
    # Extract start and end times for each cluster
    df['start_time'] = df.groupby('cluster')['date'].transform('min')
    df['end_time'] = df.groupby('cluster')['date'].transform('max')
    df['duration'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60  # in minutes
    
    return df

def plot_workout_trends_compressed(df):
    """
    Plot push-up trends over compressed time (no rest gaps),
    with annotations, grid, and session markers.
    """
    plt.figure(figsize=(14, 6))
    plt.title('Workout Timeline (Compressed)')
    plt.xlabel('Session Time')
    plt.ylabel('Push-up Count')
    plt.grid(True, linestyle='--', alpha=0.5)

    palette = sns.color_palette("husl", df['cluster'].nunique())

    cluster_offset = 0
    tick_positions = []
    tick_labels = []

    for cluster_id, group in df.groupby('cluster'):
        # Fake time index
        fake_time = list(range(cluster_offset, cluster_offset + len(group)))
        cluster_offset += len(group) + 4  # wider spacing

        # Plot lines and fill
        sns.lineplot(x=fake_time, y=group['count'], marker='o',
                     label=f"Session {cluster_id}", color=palette[cluster_id])
        plt.fill_between(fake_time, group['count'], alpha=0.2, color=palette[cluster_id])

        # 🔢 Annotate max push-up count
        max_idx = group['count'].idxmax()
        max_time = fake_time[group.index.get_loc(max_idx)]
        max_count = group.loc[max_idx, 'count']
        plt.annotate(f'Max: {max_count}', xy=(max_time, max_count),
                     xytext=(max_time, max_count + 1),
                     ha='center', fontsize=9, color=palette[cluster_id],
                     arrowprops=dict(arrowstyle='->', lw=1, color=palette[cluster_id]))

        # 🕒 X-ticks as start → end
        start_label = group['start_time'].iloc[0].strftime('%H:%M:%S')
        end_label = group['end_time'].iloc[0].strftime('%H:%M:%S')
        tick_positions.extend([fake_time[0], fake_time[-1]])
        tick_labels.extend([f"Start\n{start_label}", f"End\n{end_label}"])

        # 🕒 Annotate session duration arrow
        duration = group['duration'].iloc[0]
        duration_label = f"{int(duration)}:{int((duration % 1) * 60):02d} min"
        y_pos = (max_count // 5 + 1) * 5  # slightly above the max push-up value
        y_pos = - 2
        start_x = fake_time[0]
        end_x = fake_time[-1]

        plt.annotate(
            '', xy=(end_x, y_pos), xytext=(start_x, y_pos),
            arrowprops=dict(arrowstyle='<->', lw=1.2, color=palette[cluster_id])
        )
        plt.text((start_x + end_x) / 2, y_pos + 0.3, duration_label,
                 ha='center', fontsize=9, color=palette[cluster_id])
    
    # Auto-rescale Y-axis based on highest push-up + annotations
    y_max = df['count'].max() + 6  # enough room for arrows + text
    # annotate total push-ups
    total_pushups = df.groupby('cluster')['count'].max().sum()
    plt.annotate(f'Total Push-ups: {total_pushups}', xy=(0, y_max),
                xytext=(0, y_max + 1), ha='left', fontsize=10,
                arrowprops=dict(arrowstyle='->', lw=1.2))

    plt.ylim(0, y_max)
    plt.xticks(tick_positions, tick_labels, rotation=0, fontsize=9)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_vertical_stacked_sessions(df):
    """
    Plot each session as its own horizontal row (rep timeline).
    """
    plt.figure(figsize=(14, 6))
    plt.title('Workout Sessions (Stacked)')
    plt.xlabel('Time (compressed)')
    # plt.ylabel('Session')

    palette = sns.color_palette("husl", df['cluster'].nunique())

    cluster_offset = 0
    tick_positions = []
    tick_labels = []

    for cluster_id, group in df.groupby('cluster'):
        session_row = df['cluster'].nunique() - cluster_id  # Y axis is inverted for better layout
        fake_time = list(range(cluster_offset, cluster_offset + len(group)))
        cluster_offset += len(group) + 4

        # Plot push-ups as timeline on fixed Y
        plt.hlines(session_row, fake_time[0], fake_time[-1], color=palette[cluster_id], linewidth=2)
        plt.scatter(fake_time, [session_row] * len(group), color=palette[cluster_id], s=60, zorder=5)

        # Annotate with push-up count
        total_reps = group['count'].max()
        start_label = group['start_time'].iloc[0].strftime('%H:%M:%S')
        end_label = group['end_time'].iloc[0].strftime('%H:%M:%S')
        duration = group['duration'].iloc[0]
        duration_str = f"{int(duration)}:{int((duration % 1) * 60):02d} min"

        plt.text(fake_time[-1] + 1, session_row, f'{total_reps} reps ({duration_str})',
                 va='center', fontsize=9, color=palette[cluster_id])

        tick_positions.append(session_row)
        tick_labels.append(f"Session {cluster_id}\n{start_label} → {end_label}")

    # Adjust Y-axis
    plt.yticks(tick_positions, tick_labels)
    plt.xticks(rotation=0, fontsize=9) # remove ticks
    plt.xticks(fontsize=9)
    plt.xlim(0, cluster_offset)
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Define the path to the workout log JSON file
    file_path = 'pushup_log_20250408.json'
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        exit(1)
    
    # Load and process the workout log
    df = load_workout_log(file_path)
    df = process_workout_log(df)
    
    # Plot trends in workout data over time
    plot_workout_trends_compressed(df)
    # plot_vertical_stacked_sessions(df)
