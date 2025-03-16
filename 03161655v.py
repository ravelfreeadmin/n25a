# %% [markdown]
#  # NCAA Tournament Feature Engineering
#
#
#
#  This script processes NCAA basketball data to create features for predicting tournament outcomes. It includes data loading, feature calculation, visualization, and post-processing steps, organized into modular sections for clarity. Team names are added to intermediate tables and the final output for visual validation using data from MTeams.csv.

# %%
# Imports and Initial Setup
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import logging
import ipywidgets as widgets
from IPython.display import display

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables and validate data files
load_dotenv()
data_path = os.environ.get('DATA_PATH')
if data_path is None:
    raise ValueError("DATA_PATH not set in .env file.")
logger.info(f"Data path: {data_path}")

# %%
# Data Loading and Validation
# Check for required files
required_files = ['MRegularSeasonDetailedResults.csv', 'MTeams.csv', 'MMasseyOrdinals.csv', 
                  'MNCAATourneyDetailedResults.csv', 'MNCAATourneySeeds.csv']
for file in required_files:
    if not os.path.exists(os.path.join(data_path, file)):
        raise FileNotFoundError(f"Required file {file} not found in {data_path}")

# Load data with efficient data types
try:
    reg_season = pd.read_csv(os.path.join(data_path, 'MRegularSeasonDetailedResults.csv'), 
                             dtype={'Season': 'int32', 'TeamID': 'int32', 'WScore': 'float32'})
    teams = pd.read_csv(os.path.join(data_path, 'MTeams.csv'), dtype={'TeamID': 'int32'})
    massey = pd.read_csv(os.path.join(data_path, 'MMasseyOrdinals.csv'), dtype={'Season': 'int32', 'TeamID': 'int32'})
    tourney_results = pd.read_csv(os.path.join(data_path, 'MNCAATourneyDetailedResults.csv'), 
                                  dtype={'Season': 'int32', 'WTeamID': 'int32', 'LTeamID': 'int32'})
    tourney_seeds = pd.read_csv(os.path.join(data_path, 'MNCAATourneySeeds.csv'), dtype={'Season': 'int32', 'TeamID': 'int32'})
except Exception as e:
    logger.error(f"Failed to load data: {e}")
    raise

# Create a mapping from TeamID to TeamName for efficient lookups
team_id_to_name = teams.set_index('TeamID')['TeamName'].to_dict()

# Define seasons to process
seasons = list(range(2015, 2017))
logger.info(f"\n### Processing Seasons {seasons[0]} to {seasons[-1]} ###")

# %% [markdown]
#  # Function Definitions
#
#
#
#  The following functions handle data processing for each season, including saving intermediates, identifying tournament teams, and calculating features. Team names are added to intermediate tables before saving for visual validation.

# %%
def save_intermediate(df, filename, season, output_dir='intermediate_files'):
    """Save DataFrame to a season-specific directory."""
    os.makedirs(f'{output_dir}/season{season}', exist_ok=True)
    df.to_csv(f'{output_dir}/season{season}/{filename}.csv', index=False)

def load_intermediate(filename, season, output_dir='intermediate_files'):
    """Load DataFrame from a season-specific directory."""
    return pd.read_csv(f'{output_dir}/season{season}/{filename}.csv')

def identify_tourney_teams(season, tourney_seeds, output_dir='intermediate_files', visualize=False):
    """
    Identify teams participating in the tournament for a given season.
    
    Parameters:
        season (int): The season to process.
        tourney_seeds (pd.DataFrame): Tournament seeds data.
        output_dir (str): Directory to save intermediate files.
        visualize (bool): Whether to generate visualizations.
    
    Returns:
        np.ndarray: Array of tournament team IDs.
    """
    tourney_teams = tourney_seeds[tourney_seeds['Season'] == season]['TeamID'].unique()
    tourney_teams_df = pd.DataFrame({'TeamID': tourney_teams})
    tourney_teams_df['TeamName'] = tourney_teams_df['TeamID'].map(team_id_to_name)
    logger.info(f"Step 1: {len(tourney_teams)} tournament teams identified for Season {season}.")
    if visualize:
        plt.bar(['Tournament Teams'], [len(tourney_teams)])
        plt.title(f'Season {season}: Number of Tournament Teams')
        plt.ylabel('Count')
        plt.show()
    save_intermediate(tourney_teams_df, 'tourney_teams', season, output_dir)
    return tourney_teams

def filter_reg_season_games(season, reg_season, output_dir='intermediate_files', visualize=False):
    """
    Filter regular season games before tournament start (DayNum < 134).
    
    Parameters:
        season (int): The season to process.
        reg_season (pd.DataFrame): Regular season game data.
        output_dir (str): Directory to save intermediate files.
        visualize (bool): Whether to generate visualizations.
    
    Returns:
        pd.DataFrame: Filtered regular season game data.
    """
    reg_season_season = reg_season[(reg_season['Season'] == season) & (reg_season['DayNum'] < 134)]
    reg_season_season['WTeamName'] = reg_season_season['WTeamID'].map(team_id_to_name)
    reg_season_season['LTeamName'] = reg_season_season['LTeamID'].map(team_id_to_name)
    logger.info(f"Step 2: {len(reg_season_season)} games filtered for Season {season} (DayNum < 134).")
    if visualize:
        plt.hist(reg_season_season['DayNum'], bins=20)
        plt.title(f'Season {season}: DayNum Distribution (Pre-Tournament)')
        plt.xlabel('DayNum')
        plt.ylabel('Frequency')
        plt.show()
    save_intermediate(reg_season_season, 'reg_season_filtered', season, output_dir)
    return reg_season_season

def transform_game_data(season, reg_season_filtered, output_dir='intermediate_files', visualize=False):
    """
    Transform game data into a team-centric format.
    
    Parameters:
        season (int): The season to process.
        reg_season_filtered (pd.DataFrame): Filtered regular season game data.
        output_dir (str): Directory to save intermediate files.
        visualize (bool): Whether to generate visualizations.
    
    Returns:
        pd.DataFrame: Transformed game data with team-centric columns.
    """
    winning_team_df = reg_season_filtered.rename(columns={
        'WTeamID': 'TeamID', 'LTeamID': 'OppID', 'WScore': 'TeamScore', 'LScore': 'OppScore',
        'WLoc': 'TeamLoc', 'WFGM': 'TeamFGM', 'WFGA': 'TeamFGA', 'WFGM3': 'TeamFGM3', 'WFGA3': 'TeamFGA3',
        'WFTM': 'TeamFTM', 'WFTA': 'TeamFTA', 'WOR': 'TeamOR', 'WDR': 'TeamDR', 'WAst': 'TeamAst',
        'WTO': 'TeamTO', 'WStl': 'TeamStl', 'WBlk': 'TeamBlk', 'WPF': 'TeamPF',
        'LFGM': 'OppFGM', 'LFGA': 'OppFGA', 'LFGM3': 'OppFGM3', 'LFGA3': 'OppFGA3',
        'LFTM': 'OppFTM', 'LFTA': 'OppFTA', 'LOR': 'OppOR', 'LDR': 'OppDR', 'LAst': 'OppAst',
        'LTO': 'OppTO', 'LStl': 'OppStl', 'LBlk': 'OppBlk', 'LPF': 'OppPF'
    })
    winning_team_df['Win'] = 1

    losing_team_df = reg_season_filtered.rename(columns={
        'LTeamID': 'TeamID', 'WTeamID': 'OppID', 'LScore': 'TeamScore', 'WScore': 'OppScore',
        'LFGM': 'TeamFGM', 'LFGA': 'TeamFGA', 'LFGM3': 'TeamFGM3', 'LFGA3': 'TeamFGA3',
        'LFTM': 'TeamFTM', 'LFTA': 'TeamFTA', 'LOR': 'TeamOR', 'LDR': 'TeamDR', 'LAst': 'TeamAst',
        'LTO': 'TeamTO', 'LStl': 'TeamStl', 'LBlk': 'TeamBlk', 'LPF': 'TeamPF',
        'WFGM': 'OppFGM', 'WFGA': 'OppFGA', 'WFGM3': 'OppFGM3', 'WFGA3': 'OppFGA3',
        'WFTM': 'OppFTM', 'WFTA': 'OppFTA', 'WOR': 'OppOR', 'WDR': 'OppDR', 'WAst': 'OppAst',
        'WTO': 'OppTO', 'WStl': 'OppStl', 'WBlk': 'OppBlk', 'WPF': 'OppPF'
    })
    losing_team_df['TeamLoc'] = losing_team_df['WLoc'].map({'H': 'A', 'A': 'H', 'N': 'N'})
    losing_team_df['Win'] = 0
    losing_team_df = losing_team_df.drop(columns=['WLoc'])

    game_df = pd.concat([winning_team_df, losing_team_df], ignore_index=True)
    game_df['TeamName'] = game_df['TeamID'].map(team_id_to_name)
    game_df['OppName'] = game_df['OppID'].map(team_id_to_name)
    if visualize:
        logger.info(f"Step 3: Sample of transformed game data for Season {season} (5 rows):")
        print(game_df[['TeamID', 'TeamName', 'OppID', 'OppName', 'TeamScore', 'OppScore', 'TeamLoc', 'Win']].head())
        plt.hist(game_df['TeamScore'], bins=20, alpha=0.7, label='Team Score')
        plt.hist(game_df['OppScore'], bins=20, alpha=0.7, label='Opponent Score')
        plt.title(f'Season {season}: Team vs. Opponent Scores')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
    save_intermediate(game_df, 'game_df_transformed', season, output_dir)
    return game_df

def add_per_game_metrics(game_df, season, output_dir='intermediate_files', visualize=False):
    """
    Add per-game efficiency metrics to game data.
    
    Parameters:
        game_df (pd.DataFrame): Transformed game data.
        season (int): The season to process.
        output_dir (str): Directory to save intermediate files.
        visualize (bool): Whether to generate visualizations.
    
    Returns:
        pd.DataFrame: Game data with added efficiency metrics.
    """
    game_df['TeamFG%'] = game_df['TeamFGM'] / game_df['TeamFGA']
    game_df['Team3P%'] = game_df['TeamFGM3'] / game_df['TeamFGA3']
    game_df['TeamFT%'] = game_df['TeamFTM'] / game_df['TeamFTA']
    game_df['OppFG%'] = game_df['OppFGM'] / game_df['OppFGA']
    game_df['Opp3P%'] = game_df['OppFGM3'] / game_df['OppFGA3']
    game_df['OppFT%'] = game_df['OppFTM'] / game_df['OppFTA']
    game_df['TeamPoss'] = game_df['TeamFGA'] - game_df['TeamOR'] + game_df['TeamTO'] + 0.4 * game_df['TeamFTA']
    game_df['OppPoss'] = game_df['OppFGA'] - game_df['OppOR'] + game_df['OppTO'] + 0.4 * game_df['OppFTA']
    game_df['TeamOE'] = game_df['TeamScore'] / game_df['TeamPoss']
    game_df['TeamDE'] = game_df['OppScore'] / game_df['OppPoss']
    game_df['Team_eFG%'] = (game_df['TeamFGM'] + 0.5 * game_df['TeamFGM3']) / game_df['TeamFGA']
    game_df['TeamTORate'] = game_df['TeamTO'] / game_df['TeamPoss']
    game_df['Pace'] = (game_df['TeamPoss'] + game_df['OppPoss']) / 2
    if visualize:
        logger.info(f"Step 4: Sample with efficiency metrics for Season {season} (5 rows):")
        print(game_df[['TeamID', 'TeamName', 'TeamFG%', 'Team3P%', 'TeamFT%', 'TeamOE', 'TeamDE', 'Team_eFG%', 'TeamTORate', 'Pace']].head())
        plt.hist(game_df['TeamDE'], bins=20, alpha=0.7, label='Team DE')
        plt.title(f'Season {season}: Offensive vs. Defensive Efficiency')
        plt.xlabel('Efficiency')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
    save_intermediate(game_df, 'game_df_metrics', season, output_dir)
    return game_df

def calculate_season_averages(game_df, season, output_dir='intermediate_files', visualize=False):
    """
    Calculate season-long averages for each team.
    
    Parameters:
        game_df (pd.DataFrame): Game data with metrics.
        season (int): The season to process.
        output_dir (str): Directory to save intermediate files.
        visualize (bool): Whether to generate visualizations.
    
    Returns:
        pd.DataFrame: Team season averages.
    """
    team_averages = game_df.groupby(['Season', 'TeamID']).agg({
        'TeamScore': 'mean', 'OppScore': 'mean', 'TeamFGM': 'mean', 'TeamFGA': 'mean', 'TeamFG%': 'mean',
        'TeamFGM3': 'mean', 'TeamFGA3': 'mean', 'Team3P%': 'mean', 'TeamFTM': 'mean', 'TeamFTA': 'mean',
        'TeamFT%': 'mean', 'TeamOR': 'mean', 'TeamDR': 'mean', 'TeamAst': 'mean', 'TeamTO': 'mean',
        'TeamStl': 'mean', 'TeamBlk': 'mean', 'TeamPF': 'mean', 'OppFGM': 'mean', 'OppFGA': 'mean',
        'OppFG%': 'mean', 'OppFGM3': 'mean', 'OppFGA3': 'mean', 'Opp3P%': 'mean', 'OppFTM': 'mean',
        'OppFTA': 'mean', 'OppFT%': 'mean', 'OppOR': 'mean', 'OppDR': 'mean', 'OppAst': 'mean',
        'OppTO': 'mean', 'OppStl': 'mean', 'OppBlk': 'mean', 'OppPF': 'mean', 'TeamPoss': 'mean',
        'OppPoss': 'mean', 'TeamOE': 'mean', 'TeamDE': 'mean', 'Team_eFG%': 'mean', 'TeamTORate': 'mean', 'Pace': 'mean'
    }).reset_index()
    team_averages['TeamName'] = team_averages['TeamID'].map(team_id_to_name)
    if visualize:
        logger.info(f"Step 5: Sample team averages for Season {season} (5 rows):")
        print(team_averages[['TeamID', 'TeamName', 'TeamScore', 'OppScore', 'TeamFG%', 'TeamOE', 'Team_eFG%']].head())
        plt.hist(team_averages['TeamScore'], bins=20)
        plt.title(f'Season {season}: Average Team Score Distribution')
        plt.xlabel('Average Team Score')
        plt.ylabel('Frequency')
        plt.show()
    save_intermediate(team_averages, 'team_averages', season, output_dir)
    return team_averages

def merge_opponent_averages(game_df, team_averages, season, output_dir='intermediate_files', visualize=False):
    """
    Merge opponent season averages into game data.
    
    Parameters:
        game_df (pd.DataFrame): Game data.
        team_averages (pd.DataFrame): Team season averages.
        season (int): The season to process.
        output_dir (str): Directory to save intermediate files.
        visualize (bool): Whether to generate visualizations.
    
    Returns:
        pd.DataFrame: Game data with opponent averages.
    """
    try:
        game_df = game_df.merge(team_averages, left_on=['Season', 'OppID'], right_on=['Season', 'TeamID'], 
                                suffixes=('', '_OppAvg'))
        game_df = game_df.drop(columns=['TeamID_OppAvg'])
    except Exception as e:
        logger.error(f"Failed to merge opponent averages: {e}")
        raise
    if visualize:
        logger.info(f"Step 6: Sample with opponent averages for Season {season} (5 rows):")
        print(game_df[['TeamID', 'TeamName', 'OppID', 'OppName', 'TeamScore', 'OppScore_OppAvg']].head())
        plt.scatter(game_df['TeamScore'], game_df['OppScore_OppAvg'], alpha=0.5)
        plt.title(f'Season {season}: Team Score vs. Opponent Avg Score')
        plt.xlabel('Team Score')
        plt.ylabel('Opponent Average Score')
        plt.show()
    save_intermediate(game_df, 'game_df_opp_averages', season, output_dir)
    return game_df

def calculate_differentials(game_df, season, output_dir='intermediate_files', visualize=False):
    """
    Calculate performance differentials against opponent averages.
    
    Parameters:
        game_df (pd.DataFrame): Game data with opponent averages.
        season (int): The season to process.
        output_dir (str): Directory to save intermediate files.
        visualize (bool): Whether to generate visualizations.
    
    Returns:
        pd.DataFrame: Game data with differentials.
    """
    differentials = {
        'PointsScoredDiff': game_df['TeamScore'] - game_df['OppScore_OppAvg'],
        'PointsAllowedDiff': game_df['TeamScore_OppAvg'] - game_df['OppScore'],
        'FGMDiff': game_df['TeamFGM'] - game_df['OppFGM_OppAvg'],
        'FGADiff': game_df['TeamFGA'] - game_df['OppFGA_OppAvg'],
        'FGM3Diff': game_df['TeamFGM3'] - game_df['OppFGM3_OppAvg'],
        'FGA3Diff': game_df['TeamFGA3'] - game_df['OppFGA3_OppAvg'],
        'FTMDiff': game_df['TeamFTM'] - game_df['OppFTM_OppAvg'],
        'FTADiff': game_df['TeamFTA'] - game_df['OppFTA_OppAvg'],
        'ORDiff': game_df['TeamOR'] - game_df['OppOR_OppAvg'],
        'DRDiff': game_df['TeamDR'] - game_df['OppDR_OppAvg'],
        'AstDiff': game_df['TeamAst'] - game_df['OppAst_OppAvg'],
        'TODiff': game_df['TeamTO'] - game_df['OppTO_OppAvg'],
        'StlDiff': game_df['TeamStl'] - game_df['OppStl_OppAvg'],
        'BlkDiff': game_df['TeamBlk'] - game_df['OppBlk_OppAvg'],
        'PFDiff': game_df['TeamPF'] - game_df['OppPF_OppAvg'],
        'OE_diff': game_df['TeamOE'] - game_df['TeamDE_OppAvg'],
        'DE_diff': game_df['TeamOE_OppAvg'] - game_df['TeamDE']
    }
    for key, value in differentials.items():
        game_df[key] = value
    if visualize:
        logger.info(f"Step 7: Sample with differentials for Season {season} (5 rows):")
        print(game_df[['TeamID', 'TeamName', 'PointsScoredDiff', 'FGMDiff', 'OE_diff']].head())
        plt.hist(game_df['PointsScoredDiff'], bins=20)
        plt.title(f'Season {season}: Points Scored Differential Distribution')
        plt.xlabel('PointsScoredDiff')
        plt.ylabel('Frequency')
        plt.show()
    save_intermediate(game_df, 'game_df_differentials', season, output_dir)
    return game_df

def average_differentials(game_df, season, output_dir='intermediate_files', visualize=False):
    """
    Calculate average differentials for each team.
    
    Parameters:
        game_df (pd.DataFrame): Game data with differentials.
        season (int): The season to process.
        output_dir (str): Directory to save intermediate files.
        visualize (bool): Whether to generate visualizations.
    
    Returns:
        pd.DataFrame: Adjusted stats with average differentials.
    """
    adjusted_stats = game_df.groupby(['Season', 'TeamID']).agg({
        'PointsScoredDiff': 'mean', 'PointsAllowedDiff': 'mean', 'FGMDiff': 'mean', 'FGADiff': 'mean',
        'FGM3Diff': 'mean', 'FGA3Diff': 'mean', 'FTMDiff': 'mean', 'FTADiff': 'mean', 'ORDiff': 'mean',
        'DRDiff': 'mean', 'AstDiff': 'mean', 'TODiff': 'mean', 'StlDiff': 'mean', 'BlkDiff': 'mean',
        'PFDiff': 'mean', 'OE_diff': 'mean', 'DE_diff': 'mean'
    }).reset_index()
    adjusted_stats['TeamName'] = adjusted_stats['TeamID'].map(team_id_to_name)
    if visualize:
        logger.info(f"Step 8: Sample adjusted stats for Season {season} (5 rows):")
        print(adjusted_stats[['TeamID', 'TeamName', 'PointsScoredDiff', 'FGMDiff']].head())
        plt.hist(adjusted_stats['PointsScoredDiff'], bins=20)
        plt.title(f'Season {season}: Average Points Scored Differential')
        plt.xlabel('PointsScoredDiff')
        plt.ylabel('Frequency')
        plt.show()
    save_intermediate(adjusted_stats, 'adjusted_stats', season, output_dir)
    return adjusted_stats

def calculate_win_percentages(game_df, massey, season, output_dir='intermediate_files', visualize=False):
    """
    Calculate various win percentages for each team.
    
    Parameters:
        game_df (pd.DataFrame): Game data.
        massey (pd.DataFrame): Massey ordinals data.
        season (int): The season to process.
        output_dir (str): Directory to save intermediate files.
        visualize (bool): Whether to generate visualizations.
    
    Returns:
        pd.DataFrame: Win percentages for each team.
    """
    win_percentages = game_df.groupby(['Season', 'TeamID']).agg({'Win': 'mean'}).reset_index()
    win_percentages.rename(columns={'Win': 'allgameswinperc'}, inplace=True)
    home_win_perc = game_df[game_df['TeamLoc'] == 'H'].groupby(['Season', 'TeamID']).agg({'Win': 'mean'}).reset_index()
    home_win_perc.rename(columns={'Win': 'homewinperc'}, inplace=True)
    away_win_perc = game_df[game_df['TeamLoc'] == 'A'].groupby(['Season', 'TeamID']).agg({'Win': 'mean'}).reset_index()
    away_win_perc.rename(columns={'Win': 'awaywinperc'}, inplace=True)
    neut_win_perc = game_df[game_df['TeamLoc'] == 'N'].groupby(['Season', 'TeamID']).agg({'Win': 'mean'}).reset_index()
    neut_win_perc.rename(columns={'Win': 'neutwinperc'}, inplace=True)
    last_10_games = game_df.sort_values('DayNum').groupby(['Season', 'TeamID']).tail(10)
    last_10_win_perc = last_10_games.groupby(['Season', 'TeamID']).agg({'Win': 'mean'}).reset_index()
    last_10_win_perc.rename(columns={'Win': 'last10gameswinperc'}, inplace=True)
    latest_day = massey[(massey['Season'] == season) & (massey['RankingDayNum'] < 134)]['RankingDayNum'].max()
    sel_rankings = massey[(massey['Season'] == season) & (massey['RankingDayNum'] == latest_day) & 
                          (massey['SystemName'] == 'SEL')]
    top_100 = sel_rankings.nsmallest(100, 'OrdinalRank')['TeamID'].tolist()
    game_df['OppTop100'] = game_df['OppID'].isin(top_100)
    vs_top_100_win_perc = game_df[game_df['OppTop100']].groupby(['Season', 'TeamID']).agg({'Win': 'mean'}).reset_index()
    vs_top_100_win_perc.rename(columns={'Win': 'winpercVSseloTop100'}, inplace=True)
    win_percentages = win_percentages.merge(home_win_perc, on=['Season', 'TeamID'], how='left')
    win_percentages = win_percentages.merge(away_win_perc, on=['Season', 'TeamID'], how='left')
    win_percentages = win_percentages.merge(neut_win_perc, on=['Season', 'TeamID'], how='left')
    win_percentages = win_percentages.merge(last_10_win_perc, on=['Season', 'TeamID'], how='left')
    win_percentages = win_percentages.merge(vs_top_100_win_perc, on=['Season', 'TeamID'], how='left')
    win_percentages = win_percentages.fillna({'homewinperc': 0, 'awaywinperc': 0, 'neutwinperc': 0, 
                                              'last10gameswinperc': 0, 'winpercVSseloTop100': 0})
    win_percentages['TeamName'] = win_percentages['TeamID'].map(team_id_to_name)
    if visualize:
        logger.info(f"Step 9: Sample win percentages for Season {season} (5 rows):")
        print(win_percentages[['TeamID', 'TeamName', 'allgameswinperc', 'homewinperc', 'awaywinperc', 'neutwinperc']].head())
        sns.boxplot(data=win_percentages[['allgameswinperc', 'homewinperc', 'awaywinperc', 'neutwinperc']])
        plt.title(f'Season {season}: Win Percentage Distributions')
        plt.ylabel('Win Percentage')
        plt.show()
    save_intermediate(win_percentages, 'win_percentages', season, output_dir)
    return win_percentages

def get_elo_rating(season, massey, output_dir='intermediate_files', visualize=False):
    """
    Get ELO ratings from Massey Ordinals.
    
    Parameters:
        season (int): The season to process.
        massey (pd.DataFrame): Massey ordinals data.
        output_dir (str): Directory to save intermediate files.
        visualize (bool): Whether to generate visualizations.
    
    Returns:
        pd.DataFrame: ELO ratings for each team.
    """
    latest_day = massey[(massey['Season'] == season) & (massey['RankingDayNum'] < 134)]['RankingDayNum'].max()
    elo_rank = massey[(massey['Season'] == season) & (massey['RankingDayNum'] == latest_day) & 
                      (massey['SystemName'] == 'SEL')][['TeamID', 'OrdinalRank']]
    elo_rank.rename(columns={'OrdinalRank': 'ELOratingatstartoftourney'}, inplace=True)
    elo_rank['TeamName'] = elo_rank['TeamID'].map(team_id_to_name)
    if visualize:
        logger.info(f"Step 10: Sample ELO ratings for Season {season} (5 rows):")
        print(elo_rank[['TeamID', 'TeamName', 'ELOratingatstartoftourney']].head())
        plt.hist(elo_rank['ELOratingatstartoftourney'], bins=20)
        plt.title(f'Season {season}: ELO Rating Distribution')
        plt.xlabel('ELO Rating')
        plt.ylabel('Frequency')
        plt.show()
    save_intermediate(elo_rank, 'elo_rank', season, output_dir)
    return elo_rank

def get_tournament_seed(season, tourney_seeds, output_dir='intermediate_files', visualize=False):
    """
    Extract tournament seeds for each team.
    
    Parameters:
        season (int): The season to process.
        tourney_seeds (pd.DataFrame): Tournament seeds data.
        output_dir (str): Directory to save intermediate files.
        visualize (bool): Whether to generate visualizations.
    
    Returns:
        pd.DataFrame: Tournament seeds for each team.
    """
    seeds = tourney_seeds[tourney_seeds['Season'] == season][['TeamID', 'Seed']]
    seeds['ncaatourneyseed'] = seeds['Seed'].str.extract(r'(\d+)').astype(int)
    seeds['TeamName'] = seeds['TeamID'].map(team_id_to_name)
    if visualize:
        logger.info(f"Step 11: Sample seeds for Season {season} (5 rows):")
        print(seeds[['TeamID', 'TeamName', 'ncaatourneyseed']].head())
        plt.hist(seeds['ncaatourneyseed'], bins=16)
        plt.title(f'Season {season}: Tournament Seed Distribution')
        plt.xlabel('Seed')
        plt.ylabel('Frequency')
        plt.show()
    save_intermediate(seeds, 'seeds', season, output_dir)
    return seeds

def compile_features(season, team_averages, adjusted_stats, win_percentages, elo_rank, seeds, teams, 
                     tourney_results, tourney_teams, output_dir='intermediate_files', visualize=False):
    """
    Compile all features into a final dataset for the season.
    
    Parameters:
        season (int): The season to process.
        team_averages (pd.DataFrame): Team season averages.
        adjusted_stats (pd.DataFrame): Adjusted stats with differentials.
        win_percentages (pd.DataFrame): Win percentages.
        elo_rank (pd.DataFrame): ELO ratings.
        seeds (pd.DataFrame): Tournament seeds.
        teams (pd.DataFrame): Team information.
        tourney_results (pd.DataFrame): Tournament results.
        tourney_teams (np.ndarray): Tournament team IDs.
        output_dir (str): Directory to save intermediate files.
        visualize (bool): Whether to generate visualizations.
    
    Returns:
        pd.DataFrame: Compiled features for the season.
    """
    features_season = team_averages[team_averages['TeamID'].isin(tourney_teams)]
    features_season = features_season.merge(adjusted_stats, on=['Season', 'TeamID'], how='left')
    features_season = features_season.merge(win_percentages, on=['Season', 'TeamID'], how='left')
    features_season = features_season.merge(elo_rank, on='TeamID', how='left')
    features_season = features_season.merge(seeds[['TeamID', 'ncaatourneyseed']], on='TeamID', how='left')
    features_season = features_season.merge(teams[['TeamID', 'TeamName']], on='TeamID', how='left')
    tourney_wins = tourney_results[tourney_results['Season'] == season].groupby(['Season', 'WTeamID']).size().reset_index(name='TourneyWins')
    tourney_wins.rename(columns={'WTeamID': 'TeamID'}, inplace=True)
    features_season = features_season.merge(tourney_wins, on=['Season', 'TeamID'], how='left')
    features_season['TourneyWins'] = features_season['TourneyWins'].fillna(0).astype(int)
    feature_cols = ['Season', 'TeamID', 'TeamName', 'ncaatourneyseed', 'allgameswinperc', 'homewinperc', 
                    'awaywinperc', 'neutwinperc', 'last10gameswinperc', 'winpercVSseloTop100', 
                    'ELOratingatstartoftourney', 'TeamScore', 'OppScore', 'TeamFGM', 'TeamFGA', 'TeamFG%', 
                    'TeamFGM3', 'TeamFGA3', 'Team3P%', 'TeamFTM', 'TeamFTA', 'TeamFT%', 'TeamOR', 'TeamDR', 
                    'TeamAst', 'TeamTO', 'TeamStl', 'TeamBlk', 'TeamPF', 'OppFGM', 'OppFGA', 'OppFG%', 
                    'OppFGM3', 'OppFGA3', 'Opp3P%', 'OppFTM', 'OppFTA', 'OppFT%', 'OppOR', 'OppDR', 
                    'OppAst', 'OppTO', 'OppStl', 'OppBlk', 'OppPF', 'PointsScoredDiff', 'PointsAllowedDiff', 
                    'FGMDiff', 'FGADiff', 'FGM3Diff', 'FGA3Diff', 'FTMDiff', 'FTADiff', 'ORDiff', 'DRDiff', 
                    'AstDiff', 'TODiff', 'StlDiff', 'BlkDiff', 'PFDiff', 'OE_diff', 'DE_diff', 'TourneyWins']
    features_season = features_season[feature_cols]
    cols_to_round_whole = [
        'TeamScore', 'OppScore', 'TeamFGM', 'TeamFGA', 'TeamFGM3', 'TeamFGA3',
        'TeamFTM', 'TeamFTA', 'TeamOR', 'TeamDR', 'TeamAst', 'TeamTO', 'TeamStl',
        'TeamBlk', 'TeamPF', 'OppFGM', 'OppFGA', 'OppFGM3', 'OppFGA3', 'OppFTM',
        'OppFTA', 'OppOR', 'OppDR', 'OppAst', 'OppTO', 'OppStl', 'OppBlk', 'OppPF',
        'PointsScoredDiff', 'PointsAllowedDiff', 'FGMDiff', 'FGADiff', 'FGM3Diff',
        'FGA3Diff', 'FTMDiff', 'FTADiff', 'ORDiff', 'DRDiff', 'AstDiff', 'TODiff',
        'StlDiff', 'BlkDiff', 'PFDiff'
    ]
    cols_to_round_3dec = [
        'TeamFG%', 'Team3P%', 'TeamFT%', 'OppFG%', 'Opp3P%', 'OppFT%',
        'allgameswinperc', 'homewinperc', 'awaywinperc', 'neutwinperc',
        'last10gameswinperc', 'winpercVSseloTop100', 'OE_diff', 'DE_diff'
    ]
    for col in cols_to_round_whole:
        if col in features_season.columns:
            features_season[col] = features_season[col].round(0).astype(int)
    for col in cols_to_round_3dec:
        if col in features_season.columns:
            features_season[col] = features_season[col].round(3)
    if visualize:
        logger.info(f"Step 12: Sample final features for Season {season} (5 rows):")
        print(features_season[['TeamID', 'TeamName', 'ncaatourneyseed', 'PointsScoredDiff', 'TourneyWins']].head())
        plt.scatter(features_season['PointsScoredDiff'], features_season['TourneyWins'])
        plt.title(f'Season {season}: PointsScoredDiff vs. TourneyWins')
        plt.xlabel('PointsScoredDiff')
        plt.ylabel('Tournament Wins')
        plt.savefig(f'{output_dir}/season{season}/points_diff_vs_wins.png')
        plt.show()
        # Additional visualizations
        sns.pairplot(features_season[['PointsScoredDiff', 'ELOratingatstartoftourney', 'TourneyWins']])
        plt.savefig(f'{output_dir}/season{season}/pairplot.png')
        plt.show()
        sns.boxplot(x='TourneyWins', y='PointsScoredDiff', data=features_season)
        plt.savefig(f'{output_dir}/season{season}/boxplot_tourneywins_points_diff.png')
        plt.show()
    save_intermediate(features_season, 'features_season', season, output_dir)
    return features_season

# %% [markdown]
#  # Main Processing Function
#
#
#
#  The `process_season` function orchestrates the feature engineering steps for a single season.

# %%
def process_season(season, reg_season, tourney_seeds, teams, massey, tourney_results, 
                   output_dir='intermediate_files', visualize=False):
    """
    Process all feature engineering steps for a single season.
    
    Parameters:
        season (int): The season to process.
        reg_season (pd.DataFrame): Regular season game data.
        tourney_seeds (pd.DataFrame): Tournament seeds data.
        teams (pd.DataFrame): Team information.
        massey (pd.DataFrame): Massey ordinals data.
        tourney_results (pd.DataFrame): Tournament results.
        output_dir (str): Directory to save intermediate files.
        visualize (bool): Whether to generate visualizations.
    
    Returns:
        pd.DataFrame: Compiled features for the season.
    """
    tourney_teams = identify_tourney_teams(season, tourney_seeds, output_dir, visualize)
    reg_season_filtered = filter_reg_season_games(season, reg_season, output_dir, visualize)
    game_df = transform_game_data(season, reg_season_filtered, output_dir, visualize)
    game_df = add_per_game_metrics(game_df, season, output_dir, visualize)
    team_averages = calculate_season_averages(game_df, season, output_dir, visualize)
    game_df = merge_opponent_averages(game_df, team_averages, season, output_dir, visualize)
    game_df = calculate_differentials(game_df, season, output_dir, visualize)
    adjusted_stats = average_differentials(game_df, season, output_dir, visualize)
    win_percentages = calculate_win_percentages(game_df, massey, season, output_dir, visualize)
    elo_rank = get_elo_rating(season, massey, output_dir, visualize)
    seeds = get_tournament_seed(season, tourney_seeds, output_dir, visualize)
    features_season = compile_features(season, team_averages, adjusted_stats, win_percentages, 
                                       elo_rank, seeds, teams, tourney_results, tourney_teams, 
                                       output_dir, visualize)
    return features_season

# %% [markdown]
#  # Interactive Widget for Season Selection
#
#
#
#  Use the dropdown to select a season and process it interactively with visualizations.

# %%
season_dropdown = widgets.Dropdown(options=seasons, description='Season:')

def on_season_change(change):
    """Process a selected season interactively when dropdown value changes."""
    season = season_dropdown.value
    features_season = process_season(season, reg_season, tourney_seeds, teams, massey, 
                                    tourney_results, visualize=True)
    logger.info(f"Processed Season {season}. Sample features:")
    print(features_season.head())

season_dropdown.observe(on_season_change, names='value')
display(season_dropdown)

# %% [markdown]
#  # Process All Seasons
#
#
#
#  This section processes all specified seasons, concatenates the features, and visualizes the number of tournament teams per season.

# %%
features_list = []
tourney_teams_count = {}
for season in seasons:
    features_season = process_season(season, reg_season, tourney_seeds, teams, massey, 
                                    tourney_results, visualize=(season == seasons[0]))
    features_list.append(features_season)
    tourney_teams_count[season] = len(identify_tourney_teams(season, tourney_seeds))

# Visualize number of tournament teams per season
plt.bar(tourney_teams_count.keys(), tourney_teams_count.values())
plt.title('Number of Tournament Teams per Season')
plt.xlabel('Season')
plt.ylabel('Number of Teams')
plt.xticks(rotation=45)
plt.savefig('tourney_teams_per_season.png')
plt.show()

# Concatenate features from all seasons
features = pd.concat(features_list, ignore_index=True)
save_intermediate(features, 'features_concatenated', season='all')

# %% [markdown]
#  # Post-Processing Steps
#
#
#
#  Handle missing values, standardize features, check correlations, and save the final dataset.

# %%
# Context-specific imputation
mean_impute_cols = ['TeamFG%', 'Team3P%', 'TeamFT%', 'ELOratingatstartoftourney']
features[mean_impute_cols] = features[mean_impute_cols].fillna(features[mean_impute_cols].mean())

# Log NaN summary before final fill
nan_summary = features.isna().sum()
logger.info(f"Columns with NaN values before final fill:\n{nan_summary[nan_summary > 0]}")

# Fill remaining NaN values with zero
features = features.fillna(0)
logger.info("Step 13: NaN values handled with context-specific imputation and zero filling.")
plt.bar(nan_summary.index[:5], nan_summary.values[:5])
plt.title('NaN Counts After Imputation (First 5 Columns)')
plt.xlabel('Column')
plt.ylabel('NaN Count')
plt.xticks(rotation=45)
plt.savefig('nan_counts_after_imputation.png')
plt.show()
save_intermediate(features, 'features_nan_filled', season='all')

# Standardize numerical features
numerical_cols = [col for col in features.columns if col not in ['Season', 'TeamID', 'TeamName', 'TourneyWins']]
scaler = StandardScaler()
features[numerical_cols] = scaler.fit_transform(features[numerical_cols])
logger.info("Step 14: Sample standardized features (5 rows):")
print(features[['TeamID', 'TeamName', 'PointsScoredDiff', 'TeamFG%']].head())
plt.hist(features['PointsScoredDiff'], bins=20)
plt.title('Standardized PointsScoredDiff Across All Seasons')
plt.xlabel('Standardized PointsScoredDiff')
plt.ylabel('Frequency')
plt.savefig('standardized_points_diff.png')
plt.show()
save_intermediate(features, 'features_standardized', season='all')

# Check correlations and remove highly correlated features
corr_matrix = features[numerical_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()
corr_matrix.to_csv('corr_matrix_step15.csv')

# Remove highly correlated features (correlation > 0.9)
high_corr = corr_matrix.abs() > 0.9
upper_tri = high_corr.where(np.triu(np.ones(high_corr.shape), k=1).astype(bool))
to_drop = [col for col in upper_tri.columns if any(upper_tri[col])]
features = features.drop(columns=to_drop)
logger.info(f"Removed highly correlated features: {to_drop}")

# Save final dataset
features.to_csv('ncaa_tourney_features_2025_visualized4.csv', index=False)
logger.info("Feature engineering complete. Data saved to 'ncaa_tourney_features_2025_visualized4.csv'.")