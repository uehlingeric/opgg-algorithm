import pandas as pd
from selenium import webdriver
import time
from bs4 import BeautifulSoup
import json
import os


def setup_webdriver():
    """Sets up and returns a Chrome WebDriver with configured options."""
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    return webdriver.Chrome(options=options)


def navigate_to_url(driver, url):
    """Navigates the WebDriver to the specified URL and waits for the page to load."""
    driver.get(url)
    time.sleep(5)  # Adjust the sleep time as necessary for the page to load


def extract_html_source(driver):
    """Extracts and returns the HTML source of the current page in the WebDriver."""
    return driver.page_source


def extract_json_data(html_content):
    """Extracts JSON data from the provided HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    script_tag = soup.find(
        'script', string=lambda t: t and 'games' in t and 'data' in t)
    if script_tag:
        json_string = script_tag.string
        start = json_string.find('{')
        end = json_string.rfind('}') + 1
        json_data = json.loads(json_string[start:end])
        return json_data
    else:
        return None


def extract_games_data(json_data):
    """Extracts the 'games' section from the JSON data."""
    # Navigate to the 'games' section via 'props' and 'pageProps'
    page_props = json_data.get('props', {})
    games_data = page_props.get('pageProps', {}).get(
        'games', None).get('data', None)
    return games_data

def extract_names_data(json_data):
    """Extracts the 'names' section from the JSON data."""
    # Navigate to the 'names' section via 'props' and 'pageProps'
    page_props = json_data.get('props', {})
    games_data = page_props.get('pageProps', {}).get('data', None)
    return games_data


def parse_games_to_csv(games_data, directory):
    """Parses the 'games' data and writes specific fields to a CSV file."""
    all_games = []

    for game in games_data:
        # Skip the game if is_opscore_active is false, is_remake is true, or not Ranked Solo/Duo
        if (not game.get('is_opscore_active', False) or 
            game.get('is_remake', False) or 
            game.get('queue_info', {}).get('queue_translate') != 'Ranked Solo/Duo'):
            continue

        version = game.get('version')
        game_length = game.get('game_length_second')
        tier_info = game.get('average_tier_info', {})
        if not isinstance(tier_info, dict):
            tier_info = {}
        tier = tier_info.get('tier', 'Unknown')

        for participant in game.get('participants', []):
            summoner_info = participant.get('summoner', {})
            puuid = summoner_info.get('puuid')
            name = summoner_info.get('game_name')
            tagline = summoner_info.get('tagline')
            champ_id = participant.get('champion_id')
            side = participant.get('team_key')
            position = participant.get('position')
            stats = participant.get('stats', {})

            all_games.append([
                stats.get('op_score'),
                stats.get('result'),
                name,
                tagline,
                champ_id,
                side,
                position,
                game_length,
                stats.get('champion_level'),
                stats.get('damage_self_mitigated'),
                stats.get('damage_dealt_to_objectives'),
                stats.get('damage_dealt_to_turrets'),
                stats.get('magic_damage_dealt_player'),
                stats.get('physical_damage_taken'),
                stats.get('physical_damage_dealt_to_champions'),
                stats.get('total_damage_taken'),
                stats.get('total_damage_dealt'),
                stats.get('total_damage_dealt_to_champions'),
                stats.get('time_ccing_others'),
                stats.get('vision_score'),
                stats.get('vision_wards_bought_in_game'),
                stats.get('sight_wards_bought_in_game'),
                stats.get('ward_kill'),
                stats.get('ward_place'),
                stats.get('turret_kill'),
                stats.get('barrack_kill'),
                stats.get('kill'),
                stats.get('death'),
                stats.get('assist'),
                stats.get('largest_multi_kill'),
                stats.get('largest_killing_spree'),
                stats.get('minion_kill'),
                stats.get('neutral_minion_kill_team_jungle'),
                stats.get('neutral_minion_kill_enemy_jungle'),
                stats.get('neutral_minion_kill'),
                stats.get('gold_earned'),
                stats.get('total_heal'),
                tier,
                game['id'],
                puuid,
                version
            ])

    df = pd.DataFrame(all_games, columns=[
        'op_score', 'result', 'name', 'tagline', 'champ_id', 'side', 'position', 'game_length',
        'champion_level', 'damage_self_mitigated', 'damage_dealt_to_objectives', 'damage_dealt_to_turrets',
        'magic_damage_dealt_player', 'physical_damage_taken', 'physical_damage_dealt_to_champions',
        'total_damage_taken', 'total_damage_dealt', 'total_damage_dealt_to_champions',
        'time_ccing_others', 'vision_score', 'vision_wards_bought_in_game', 'sight_wards_bought_in_game',
        'ward_kill', 'ward_place', 'turret_kill', 'barrack_kill', 'kill', 'death', 'assist',
        'largest_multi_kill', 'largest_killing_spree', 'minion_kill', 'neutral_minion_kill_team_jungle',
        'neutral_minion_kill_enemy_jungle', 'neutral_minion_kill', 'gold_earned', 'total_heal',
        'tier', 'game_id', 'puuid', 'version'
    ])


    # Define CSV file path
    os.makedirs(directory, exist_ok=True)
    csv_file = os.path.join(directory, 'games.csv')

    export_data(df, csv_file)


def export_data(new_data_df, filename):
    """Append new data to an existing CSV file, or overwrite it if columns are different or an error occurs."""
    # Check if the file exists
    if os.path.exists(filename):
        try:
            # Read the existing file's first line to compare columns
            existing_columns = pd.read_csv(filename, nrows=0).columns

            # Compare columns, overwrite if they are different
            if not new_data_df.columns.equals(existing_columns):
                new_data_df.to_csv(filename, header=True, index=False)
            else:
                # Append data without header
                new_data_df.to_csv(filename, mode='a',
                                   header=False, index=False)

        except Exception as e:
            # In case of an error, overwrite the file
            new_data_df.to_csv(filename, header=True, index=False)

    else:
        # Write new file with header
        new_data_df.to_csv(filename, header=True, index=False)

def names_to_dataframe(json_data):
    """Converts JSON data to a pandas DataFrame containing game names."""
    game_names = []

    # Iterate through the JSON data and extract 'game_name' from each summoner
    for item in json_data:
        if 'summoner' in item and 'game_name' in item['summoner'] and 'tagline' in item['summoner']:
            game_names.append(item['summoner']['game_name'] + '-' + item['summoner']['tagline'])

    # Create DataFrame
    df = pd.DataFrame(game_names, columns=['name'])

    return df


def main():
    """Main function to orchestrate the scraping process."""
    driver = setup_webdriver()

    # URL for the leaderboard
    leaderboard_url = 'https://www.op.gg/leaderboards/tier?region=na&tier=master&page=1'

    # Navigate to the leaderboard and extract summoner names
    navigate_to_url(driver, leaderboard_url)
    leaderboard_html = extract_html_source(driver)
    leaderboard_json = extract_json_data(leaderboard_html)

    # Extract 'names' data
    names_data = extract_names_data(leaderboard_json)
    summoner_names_df = names_to_dataframe(names_data)
    summoner_names = summoner_names_df['name'].tolist()

    all_summoners_data = []

    for name in summoner_names:
        try:
            # Construct the summoner URL and process their data
            summoner_url = f'https://www.op.gg/summoners/kr/{name}'
            navigate_to_url(driver, summoner_url)
            summoner_html = extract_html_source(driver)
            summoner_json = extract_json_data(summoner_html)

            # Extract 'games' data
            games_data = extract_games_data(summoner_json)
            if games_data:
                all_summoners_data.extend(games_data)

        except Exception as e:
            print(f"Error processing summoner {name}: {e}")
            # Export data gathered so far before continuing with the next summoner
            if all_summoners_data:
                parse_games_to_csv(all_summoners_data, '../data/raw')
                all_summoners_data = []  # Reset the list after exporting

    driver.close()

    # Export any remaining data after processing all summoners
    if all_summoners_data:
        parse_games_to_csv(all_summoners_data, '../data/raw')
        print("All games data parsed and written to '../data/raw/games.csv'")
    else:
        print("No games data found for any summoner.")

if __name__ == "__main__":
    main()
