import subprocess

# Create oscar_winners_1929_2016.csv
script_directory = "oscar_scrapper"

command = [
    "python",
    "oscar_scrapper.py",
    "--years_interval",
    "1929",
    "2016",
    "--output_file",
    "../../../data/oscar_winners_1929_2016.csv",
]

subprocess.run(command, cwd=script_directory)

# Create oscar_movies_all_categories.csv
command = [
    "python",
    "dataset_creation.py",
    "--n_movies",
    "0",
    "--oscars_out",
    "../../data/oscar_movies_all_categories.csv",
]


# Create oscar_movies.csv and all_other_movies.csv
command = [
    "python",
    "dataset_creation.py",
    "--n_movies",
    "100000",
    "--others_out",
    "../../data/all_other_movies.csv" "--categories",
    "best motion picture",
    "best picture",
]

subprocess.run(command)

# Create other_awards.csv
script_directory = "other_awards"

command = [
    "python",
    "create_dataset_others.py",
]

subprocess.run(command, cwd=script_directory)

# Create review dataset
script_directory = "reviews_scraper"

command = [
    "python",
    "reviews_scraper.py",
    "--output_directory",
    "../../../data/imdb_reviews/scraped_reviews",
    "--number_years_from_release",
    "2",
    "--input_dataset_path",
    "../../../data/oscar_movies.csv",
    "--imdb_id_column",
    "tconst",
    "--release_year_column",
    "release",
]

subprocess.run(command, cwd=script_directory)

command = [
    "python",
    "combine_reviews.py",
    "--input_dataset_path",
    "../../../data/oscar_movies.csv",
    "--imdb_id_column",
    "tconst",
    "--input_reviews_directory",
    "../../../data/imdb_reviews/scraped_reviews",
    "--output_dataset_path",
    "../../../data/imdb_reviews/imdb_reviews_best_picture_2years_from_release.csv",
]

subprocess.run(command, cwd=script_directory)

command = [
    "python",
    "compute_compound_scores.py",
]

subprocess.run(command, cwd=script_directory)
