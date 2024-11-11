import pandas as pd
from bs4 import BeautifulSoup
from argparse import ArgumentParser
from datetime import datetime
import logging

from selenium import webdriver
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser(description="Scrape Oscar data")

    parser.add_argument(
        "--base_url",
        default="https://www.oscars.org/oscars/ceremonies",
        type=str,
        help="Base URL to the Oscar ceremony winners page",
    )

    parser.add_argument(
        "--years_interval",
        default=None,
        nargs=2,
        type=int,
        help="Years interval to scrape (inclusive)",
    )

    parser.add_argument(
        "--oscar_categories",
        default=None,
        nargs="+",
        type=str,
        help="Oscar categories to scrape, if None, scrape all",
    ),

    parser.add_argument(
        "--output_file",
        default="oscar_winners.csv",
        type=str,
        help="Output file to save the scraped data",
    )

    args = parser.parse_args()

    return args


"""
    Get the winner and nominees of a movie from the soup of the oscar item

    Args:
        oscar_soup: BeautifulSoup: The soup of the oscar item
        movie_name_class: str: The class of the movie name field

    Returns:
        winner: str: The winner of the category
        nominees: list: The nominees of the category
"""


def parse_movie_winner_nominees(
    oscar_soup: BeautifulSoup,
    movie_name_class="field--name-field-award-film",
):
    winner_nominnes_soup = oscar_soup.find_all("div", class_=movie_name_class)
    if len(winner_nominnes_soup) == 0:
        return None, None

    # The first element is the winner, the rest are the nominees
    winner = winner_nominnes_soup[0].text.lower()
    nominees = [nominee.text.lower() for nominee in winner_nominnes_soup[1:]]

    # Remove new lines characters
    winner = winner.replace("\n", "")
    nominees = [nominee.replace("\n", "") for nominee in nominees]

    return winner, nominees


def parse_oscar_category(
    oscar_soup: BeautifulSoup,
    category_class="field--name-field-award-category-oscars",
) -> str:
    category_soup = oscar_soup.find("div", class_=category_class)
    return category_soup.text.lower()


# For some categories, the oscars page does not follow the same structure
# as the other categories, so we need to handle them separately
SPECIAL_CATEGORIES = ["international feature film"]


def parse_movie_winner_nominees_special_category(
    oscar_soup: BeautifulSoup,
    category: str,
):
    if category == SPECIAL_CATEGORIES[0]:
        # Here the usual winner field contains the corresponding country
        # of the movie, and the winner is in the entity field (it is reversed)
        return parse_movie_winner_nominees(
            oscar_soup, movie_name_class="field--name-field-award-entities"
        )

    else:
        raise ValueError(f"Category {category} is not a special category")


"""
  Get the oscar ceremony date from the oscars page

  Args:
    page_source: str: The page source of the oscars page
"""


def scrape_ceremony_date(
    page_source: str,
    ceremony_date_class="field--name-field-date-time",
):
    # Parse the page
    soup = BeautifulSoup(page_source, "html.parser")

    # Get the date of the ceremony (in a string format)
    ceremony_date_string = soup.find("div", class_=ceremony_date_class).text.strip()

    # Cast the date to a "datetime" object
    ceremony_date = datetime.strptime(ceremony_date_string, "%A, %B %d, %Y")

    return ceremony_date


"""
  Get the oscar winners and nominees from the oscars page

  Args:
    page_source: str: The page source of the oscars page
    oscar_categories: list: The oscar categories to scrape, if None, scrape all
"""


def scrape_winners_nominees(
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

    oscar_winners = []
    oscar_nominees = []
    for oscar_soup in oscars_soup_list:
        category = parse_oscar_category(oscar_soup)

        # Skip if the category is not in the list
        if oscar_categories is not None and category not in oscar_categories:
            continue

        # Try to parse the winner
        if category in SPECIAL_CATEGORIES:
            winner, nominees = parse_movie_winner_nominees_special_category(
                oscar_soup, category
            )
        else:
            winner, nominees = parse_movie_winner_nominees(oscar_soup)

        if winner is None or nominees is None:
            logging.warning("Unable to parse the winner for the category: %s", category)
            continue

        oscar_winners.append((category, winner))
        for nominee in nominees:
            oscar_nominees.append((category, nominee))

    return oscar_winners, oscar_nominees


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
    oscar_winners, oscar_nominees = scrape_winners_nominees(
        page_source, oscar_categories
    )
    ceremony_date = scrape_ceremony_date(page_source)

    MOVIE_NAME_COLUMN = "movie_name"
    OSCAR_CATEGORY_COLUMN = "oscar_category"
    YEAR_COLUMN = "year"
    CEREMONY_DATE_COLUMN = "ceremony_date"
    WINNER_COLUMN = (
        "winner"  # True if the movie is the winner, False if it is a nominee
    )

    data = []

    # Add the winners
    for category, winner in oscar_winners:
        data.append(
            {
                MOVIE_NAME_COLUMN: winner,
                OSCAR_CATEGORY_COLUMN: category,
                YEAR_COLUMN: year,
                CEREMONY_DATE_COLUMN: ceremony_date,
                WINNER_COLUMN: True,
            }
        )

    # Add the nominees
    for category, nominee in oscar_nominees:
        data.append(
            {
                MOVIE_NAME_COLUMN: nominee,
                OSCAR_CATEGORY_COLUMN: category,
                YEAR_COLUMN: year,
                CEREMONY_DATE_COLUMN: ceremony_date,
                WINNER_COLUMN: False,
            }
        )

    return pd.DataFrame(data)


def main():
    args = parse_args()

    # Create the driver
    driver = create_driver()

    base_url = args.base_url
    categories = args.oscar_categories
    years = range(args.years_interval[0], args.years_interval[1] + 1)
    output_file = args.output_file

    dfs = []

    loading_bar = tqdm(total=len(years), desc="Scraping oscars")

    for year in years:
        df = scrape_year(driver, base_url, year, categories)
        dfs.append(df)

        loading_bar.update(1)

    loading_bar.close()
    driver.quit()

    # Concatenate the dataframes
    final_df = pd.concat(dfs).reset_index(drop=True)

    # Save the data
    final_df.to_csv(output_file, index=False)

    pass


if __name__ == "__main__":
    main()
