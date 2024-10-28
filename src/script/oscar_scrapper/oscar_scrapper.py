import pandas as pd
import requests
from bs4 import BeautifulSoup
from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description="Scrape Oscar data")

    parser.add_argument(
        "base_url",
        default="https://www.oscars.org/oscars/ceremonies",
        type=str,
        help="Base URL to the Oscar ceremony winners page",
    )

    parser.add_argument(
        "years_interval",
        default=None,
        type=tuple,
        help="Years interval to scrape",
    )

    parser.add_argument(
        "movie_oscar_categories",
        default=None,
        type=list,
        help="Oscar given to a movie categories to scrape (ex: Best picture, ...)",
    ),

    parser.add_argument(
        "person_oscar_categories",
        default=None,
        type=list,
        help="Oscar given to a person categories to scrape, (ex: Best actor, ...)",
    )

    parser.add_argument(
        "output_file",
        default="oscar_winners.csv",
        type=str,
        help="Output file to save the scraped data",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
