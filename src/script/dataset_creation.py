import pandas as pd


def main():
    PATH = "../../data/"

    # Importing the datasets
    oscar_winners = pd.read_csv(PATH + "oscar_winners_1929_2016.csv")
    titles = pd.read_csv(PATH + 'title.basics.tsv',sep='\t')
    ratings = pd.read_csv(PATH + 'title.ratings.tsv',sep='\t')
    col_names = ['wiki mID', 'freebase mID', 'name', 'release','revenue','runtime','languages','countries','genres' ]
    metadata = pd.read_csv(PATH + 'MovieSummaries/movie.metadata.tsv',sep='\t', names = col_names, header = None)

    # Creating a df with the titles and their ratings
    ratings_title = pd.merge(titles, ratings, on = 'tconst')
    ratings_title['primaryTitle'] = ratings_title['primaryTitle'].str.lower()
    
    # Inner join on the oscar and the movies, keeping only the ones nominated and winners
    df = pd.merge(ratings_title, oscar_winners, left_on = 'primaryTitle', right_on = 'movie_name', how = 'inner')

    # Dropping unimportant columns
    df = df.drop(columns=['isAdult','endYear', 'movie_name', 'originalTitle'])

    # As the runtime is a string
    df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'], errors = 'coerce')

    metadata['name'] = metadata['name'].str.lower()

    # Merging the CMU dataset with the IMDB one 
    df = pd.merge(df, metadata, left_on  = 'primaryTitle', right_on = 'name')

    # Renaming for better understanding
    df = df.rename(columns = {'year':'oscar_year', 'genres_x':'IMDB_genres', 'genres_y': 'CMU_genres'})


    # Dropping unimportant columns
    df = df.drop(columns=['titleType', 'wiki mID', 'name', 'freebase mID', 'languages'])

    # Keeping only the categories of best picture and best motion picture 
    oscars_picture = df[(df['oscar_category'] == 'best motion picture') | (df['oscar_category'] == 'best picture')]
    oscars_picture = oscars_picture.reset_index(drop=True)

    # Extracting the 4 digit year to have a coherent column and converting them to numeric
    oscars_picture['release'] = oscars_picture['release'].str.extract(r'(\d{4})')
    oscars_picture['release'] = pd.to_numeric(oscars_picture['release'], errors = 'coerce')
    oscars_picture['startYear'] = pd.to_numeric(oscars_picture['startYear'], errors = 'coerce')

    # Realigning with the correct movies: as several movies may have the same name
    # Realign with the date, the movies must be released the year before the ceremony
    oscars_picture = oscars_picture[oscars_picture['release'] == oscars_picture['oscar_year'] - 1] 
    oscars_picture = oscars_picture[oscars_picture['startYear'] == oscars_picture['release']]

    # Realign with the runtime (with a +/- 5 min interval)
    oscars_picture = oscars_picture[(oscars_picture['runtimeMinutes'] >= oscars_picture['runtime'] - 5) &
                                    (oscars_picture['runtimeMinutes'] <= oscars_picture['runtime'] + 5)] 

    # Remove movie with small number of ratings
    oscars_picture = oscars_picture[oscars_picture['numVotes'] > 10]

    oscars_picture = oscars_picture.reset_index(drop=True)

    # Final cleaning
    oscars_picture = oscars_picture.drop(columns = ['release', 'runtimeMinutes'])
    oscars_picture = oscars_picture.rename(columns = {'startYear':'release'})

    # Exporting to a csv
    oscars_picture.to_csv(PATH + 'final_data.csv', index = False)





if __name__ == "__main__":
    main()
