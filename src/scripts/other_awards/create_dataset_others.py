import pandas as pd

"""
Creates the dataset which contains movies' data, with Bafta and Golden Globes stats. The movies kept are only those appearing in all 3 datasets 
(for milestone 3: do another merge)
"""
# Importing all the datasets
PATH = '../../../data/'
oscars = pd.read_csv(PATH + 'oscar_movies.csv')
PATH = PATH + 'other_awards/'
bafta = pd.read_csv(PATH + 'bafta_films.csv')
bafta = bafta.drop(columns='workers')
bafta['nominee'] = bafta['nominee'].str.lower()

golden_globe = pd.read_csv(PATH + 'golden_globe_awards.csv')
golden_globe = golden_globe.drop(columns = ['ceremony', 'film'])
golden_globe['nominee'] = golden_globe['nominee'].str.lower()

# Cleaning the categories
bafta['category'] = bafta['category'].str.replace(r' in \d{4}', '', regex=True)
bafta['category'] = bafta['category'].str.replace(r'^Film \| ', '', regex=True)

# Keeping the "best film" category and best motion picture
bafta_best_film = bafta[(bafta['category'] == "Best Film") | 
                        (bafta['category'] == "Film From Any Source")|
                        (bafta['category'] == "Film")]
bafta_best_film = bafta_best_film.reset_index(drop = True)

gg_motion_picture = golden_globe[golden_globe['category'].str.startswith("Best Motion Picture -")]
gg_motion_picture = gg_motion_picture.reset_index(drop = True)

# Merge Golden globes and Bafta
merged_df = pd.merge(gg_motion_picture, bafta_best_film, on = 'nominee', how = 'outer')
# Replace NaN values with False in the winner columns
merged_df['win'] = merged_df['win'].fillna(False)
merged_df['winner'] = merged_df['winner'].fillna(False)

merged_df = merged_df.rename(columns = {'year_award':'year_gg', 'category_x': 'category_gg', 'win':'win_gg', 'year':'year_b', 'category_y':'category_b', 'winner':'win_b'})

all_merged = pd.merge(oscars, merged_df, left_on = 'primaryTitle', right_on = 'nominee', how = 'inner')
all_merged = all_merged.drop(columns = ['year_film', 'nominee'])
all_merged = all_merged.rename(columns = {'winner':'win_o'})


all_merged.to_csv(PATH + "other_awards.csv")