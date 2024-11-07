import pandas as pd
from argparse import ArgumentParser
import logging
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = ArgumentParser(description="Scrape IMDb Reviews")

    parser.add_argument(
        "--input_dataset_path",
        required=True,
        type=str,
        help="Path to the input dataset from which the reviews have been scraped",
    )

    parser.add_argument(
        "--imdb_id_column",
        default="imdb_id",
        type=str,
        help="Name of the column containing the IMDb ID",
    )

    parser.add_argument(
        "--input_reviews_directory",
        required=True,
        type=str,
        help="Directory containing the reviews scraped",
    )

    parser.add_argument(
        "--output_dataset_path",
        default="imdb_reviews.csv",
        type=str,
        help="Path to the output dataset",
    )

    args = parser.parse_args()
    return args


def main(args):
    logging.info("Reading the input dataset")
    input_df = pd.read_csv(args.input_dataset_path)

    logging.info("Reading the reviews")
    reviews_dfs = []
    for imdb_id in tqdm(input_df[args.imdb_id_column]):
        reviews_df = pd.read_csv(
            os.path.join(args.input_reviews_directory, f"{imdb_id}_reviews.csv")
        )

        reviews_dfs.append(reviews_df)

    logging.info("Combining the reviews")
    reviews_df = pd.concat(reviews_dfs)

    logging.info("Saving the combined reviews")
    reviews_df.to_csv(args.output_dataset_path, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
