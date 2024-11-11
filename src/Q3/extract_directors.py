#gets director name fom IMDB
import pandas as pd



#loads datasets to merge:
df_director = pd.read_csv('../../data/title.crew.tsv', sep='\t')
df_name = pd.read_csv('../../data/name.basics.tsv', sep='\t')
df_title = pd.read_csv('../../data/title.basics.tsv', sep = '\t')


df_merged = pd.merge(df_title, df_director, on='tconst', how='inner')

df_merged = df_merged[df_merged['directors'] != '\\N']


nconsts = df_name['nconst'].values
dir_names = df_name['primaryName'].values
const_dir_mapping = mapping = dict(zip(nconsts, dir_names))

def nconst_to_name(const):
    if ',' in const:
        return[const_dir_mapping[cst] for cst in const.split(',')]
    else:
        return const_dir_mapping[const]
    


df_merged['directors'] = df_merged['directors'].apply(nconst_to_name)


#CMU dataset to get revenues to estimate the popularity of a director:
df_movies = pd.read_csv('../../data/MovieSummaries/movie.metadata.tsv', sep='\t', header=None)
# Add column names
df_movies.columns = ["wikipedia_ID", "freebase_ID", "primaryTitle", "release_date", "box_office_revenue", "runtime", "languages", "countries", "genres"]
#drop missing values
df_movies = df_movies.dropna(subset=['box_office_revenue'])


df_director_movie = pd.merge(df_movies, df_merged, on='primaryTitle', how='inner')


#drop columns presents in both df_merged and df_movies and useless columns:
df_director_movie.drop(columns=['wikipedia_ID', 'freebase_ID', 'writers', 'runtime', 'genres_x', 'startYear', 'endYear'], inplace=True)


#keeps movies only:
df_director_movie = df_director_movie[df_director_movie['titleType'] == 'movie']

#remove duplicate in column name:
df_director_movie = df_director_movie.rename(columns={'genres_y': 'genres'})



#save df for drive
df_director_movie.to_csv('../../data/movie_directors.csv')


