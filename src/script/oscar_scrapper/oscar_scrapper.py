import pandas as pd
import requests
from bs4 import BeautifulSoup
from argparse import ArgumentParser
from typing import Literal
import logging

from selenium import webdriver
from selenium.webdriver.firefox.options import Options


def parse_args():
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
        help="Years interval to scrape, (for example, (2010, 2020))",
    )

    parser.add_argument(
        "oscar_categories",
        default=None,
        type=list[str],
        help="Oscar categories to scrape, if None, scrape all",
    ),

    parser.add_argument(
        "output_file",
        default="oscar_winners.csv",
        type=str,
        help="Output file to save the scraped data",
    )

    args = parser.parse_args()

    # Check the arguments are valid
    if args.years_interval is not None and len(args.years_interval) != 2:
        raise ValueError(
            "The years_interval argument should be a tuple with two elements"
        )

    return args


def parse_movie_winner(
    oscar_soup: BeautifulSoup,
    winner_movie_class="field--name-field-award-film",
) -> str:
    winner_movie_soup = oscar_soup.find("div", class_=winner_movie_class)

    return winner_movie_soup.text.lower() if winner_movie_soup is not None else None


def parse_oscar_category(
    oscar_soup: BeautifulSoup,
    category_class="field--name-field-award-category-oscars",
) -> str:
    category_soup = oscar_soup.find("div", class_=category_class)
    return category_soup.text.lower()


def scrape_winners(
    page_source: str,
    oscar_categories: list | None = None,
):
    # Parse the page
    soup = BeautifulSoup(page_source, "html.parser")
    oscars_pane_soup = soup.find("article", class_="oscars").find(
        "div", class_="field--name-field-award-categories"
    )
    oscars_soup_list = oscars_pane_soup.find_all(
        "div", class_="field__item", recursive=False
    )

    oscar_winners = dict()
    for oscar_soup in oscars_soup_list:
        category = parse_oscar_category(oscar_soup)

        # Skip if the category is not in the list
        if oscar_categories is not None and category not in oscar_categories:
            continue

        # Try to parse the winner
        winner = parse_movie_winner(oscar_soup)
        if winner is None:
            logging.warning("Unable to parse the winner for the category: %s", category)
            continue

        oscar_winners[category] = winner

    return oscar_winners


def create_driver() -> webdriver.Firefox:
    options = webdriver.FirefoxOptions()
    options.add_argument("--headless")

    driver = webdriver.Firefox(options=options)
    return driver


def get_page_source(
    driver: webdriver.Firefox,
    base_url: str,
    year: int,
) -> str:
    url = f"{base_url}/{year}"
    driver.get(url)

    return driver.page_source


def scrape_year(
    driver: webdriver.Firefox,
    base_url: str,
    year: int,
    oscar_categories: list | None = None,
) -> pd.DataFrame:
    page_source = get_page_source(driver, base_url, year)
    oscar_winners = scrape_winners(page_source, oscar_categories)

    MOVIE_NAME_COLUMN = "movie_name"
    OSCAR_CATEGORY_COLUMN = "oscar_category"
    YEAR_COLUMN = "year"

    data = []

    for category, winner in oscar_winners.items():
        data.append(
            {
                MOVIE_NAME_COLUMN: winner,
                OSCAR_CATEGORY_COLUMN: category,
                YEAR_COLUMN: year,
            }
        )

    return pd.DataFrame(data)


def main():
    pass


if __name__ == "__main__":
    main()
