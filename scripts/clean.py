"""
Eric Uehling
4.30.24

Description: Clean the data into a single CSV file.
"""

import pandas as pd
import os


def calculate_combined_metrics(data):
    """Calculate kill participation and percentage metrics for groups with exactly 5 players."""
    # Group by game_id and side
    grouped = data.groupby(['game_id', 'side'])

    # Filter out groups that don't have exactly 5 players
    valid_groups = grouped.filter(lambda x: len(x) == 5)

    # Calculate total metrics
    totals = valid_groups.groupby(['game_id', 'side']).agg({
        'kill': 'sum',
        'gold_earned': 'sum',
        'total_damage_dealt_to_champions': 'sum',
        'total_damage_taken': 'sum'
    }).reset_index().rename(columns={
        'kill': 'total_kills',
        'gold_earned': 'total_gold',
        'total_damage_dealt_to_champions': 'total_dmg',
        'total_damage_taken': 'total_dmg_taken'
    })

    # Merge with the original dataframe
    valid_groups = pd.merge(valid_groups, totals, on=['game_id', 'side'])

    # Calculate the percentages
    valid_groups['kp'] = (
        (valid_groups['kill'] + valid_groups['assist']) / valid_groups['total_kills']).round(3)
    valid_groups.loc[valid_groups['total_kills'] == 0, 'kp'] = 0
    valid_groups['gold_perc'] = (
        valid_groups['gold_earned'] / valid_groups['total_gold']).round(3)
    valid_groups['dmg_perc'] = (
        valid_groups['total_damage_dealt_to_champions'] / valid_groups['total_dmg']).round(3)
    valid_groups['dmg_taken_perc'] = (
        valid_groups['total_damage_taken'] / valid_groups['total_dmg_taken']).round(3)

    return valid_groups


def calculate_diff(data):
    """Calculate difference metrics for numeric columns for groups with exactly 2 players."""
    # Define the numeric columns for differential calculations
    numeric_cols_for_diff = {
        'cs': 'cs_diff',
        'gold_earned': 'gold_diff',
        'champion_level': 'level_diff',
        'total_damage_taken': 'dmg_taken_diff',
        'total_damage_dealt_to_champions': 'dmg_diff'
    }

    # Initialize differential columns with zeros
    for diff_col in numeric_cols_for_diff.values():
        data[diff_col] = 0

    # Group by game_id and position
    grouped = data.groupby(['game_id', 'position'])

    # Filter out groups that don't have exactly 2 players
    valid_groups = grouped.filter(lambda x: len(x) == 2)

    # Recalculate groupby on filtered data
    re_grouped = valid_groups.groupby(['game_id', 'position'])

    # Calculate differentials for valid groups
    for _, group in re_grouped:
        for orig_col, diff_col in numeric_cols_for_diff.items():
            diff = group[orig_col].diff().iloc[1]
            valid_groups.loc[group.index[0], diff_col] = -diff
            valid_groups.loc[group.index[1], diff_col] = diff

    return valid_groups


def calculate_metrics(data):
    """Calculate additional metrics for the dataset."""
    # KDA and CS (Creep Score)
    data['kda'] = ((data['kill'] + data['assist']) /
                   data['death'].replace(0, 1)).round(3)
    data['cs'] = data['minion_kill'] + data['neutral_minion_kill']

    # Calculate combined kill participation and percentage metrics
    data = calculate_combined_metrics(data)

    # Calculate differentials
    data = calculate_diff(data)

    return data


def process_game_data(data):
    """Process game data to include required metrics."""
    data['length'] = (data['game_length'] / 60).round(3)  # Convert to minutes

    data = calculate_metrics(data)
    data = process_champ_ids(data)

    # Convert 'result' to 0 for 'LOSE' and 1 for 'WIN'
    data['win'] = data['result'].apply(lambda x: 1 if x == 'WIN' else 0)

    # Rename columns as per desired output
    column_mapping = {
        'total_damage_dealt_to_champions': 'dmg',
        'total_damage_taken': 'dmg_taken',
        'vision_score': 'vision',
        'gold_earned': 'gold',
        'champion_level': 'level',
        'damage_self_mitigated': 'mitigated_dmg',
        'damage_dealt_to_objectives': 'objective_dmg',
        'damage_dealt_to_turrets': 'turret_dmg',
        'magic_damage_dealt_player': 'magic_dmg',
        'physical_damage_taken': 'ad_dmg_taken',
        'physical_damage_dealt_to_champions': 'ad_dmg',
        'total_damage_dealt': 'all_dmg',
        'time_ccing_others': 'cc_score',
        'vision_wards_bought_in_game': 'pinks_bought',
        'barrack_kill': 'inhib_kill',
        'largest_killing_spree': 'largest_kill_spree',
    }
    data = data.rename(columns=column_mapping)

    data['dmg_per_gold'] = (data['dmg'] / data['gold']).round(3)

    # Define columns to keep as per desired output
    columns_to_keep = ['champ', 'position', 'op_score', 'win', 'length', 
                       'kill', 'death', 'assist', 'kda',
                       'dmg', 'magic_dmg', 'ad_dmg', 'all_dmg', 
                       'dmg_taken', 'ad_dmg_taken', 'mitigated_dmg', 'total_heal', 
                       'cs', 'gold', 'level',
                       'kp', 'dmg_perc', 'dmg_taken_perc', 'gold_perc',
                       'turret_kill', 'inhib_kill', 'objective_dmg', 'turret_dmg', 
                       'largest_multi_kill', 'largest_kill_spree',
                       'cc_score',
                       'dmg_per_gold',
                       'vision', 'pinks_bought', 'ward_kill', 'ward_place',
                       'cs_diff', 'gold_diff', 'level_diff', 'dmg_taken_diff', 'dmg_diff']
    return data[columns_to_keep]

def process_champ_ids(data):
    """
    Read in champions.csv and change champ_id column to champ, with the new value 
    being the champion name instead of the key value.
    """
    # Load the champions data
    champions_data = pd.read_csv('../data/processed/champions.csv')
    
    # Merge the original data with the champions data
    # Now using 'champ_id' in 'data' which corresponds to 'key' in champions_data
    data = pd.merge(data, champions_data[['key', 'champ_name']], left_on='champ_id', right_on='key', how='left')
    
    # Drop the old 'champ_id' column and 'key' column from the merged data
    data = data.drop(columns=['champ_id', 'key'])
    
    # Rename 'champ_name' to 'champ'
    data = data.rename(columns={'champ_name': 'champ'})
    
    return data


def main():
    raw_data_file = '../data/raw/games.csv'
    processed_data_dir = '../data/processed'

    data = pd.read_csv(raw_data_file)

    # Remove duplicates
    data = data.drop_duplicates()

    processed_data = process_game_data(data)

    # Drop rows with length less than 15
    processed_data = processed_data[processed_data['length'] >= 15]

    # Drop rows with missing values
    # processed_data = processed_data.dropna()

    processed_data.to_csv(os.path.join(processed_data_dir, 'games.csv'), index=False)

if __name__ == "__main__":
    main()
