import pandas as pd
from bs4 import BeautifulSoup
from argparse import ArgumentParser
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


def parse_movie_winner_nominees(
    oscar_soup: BeautifulSoup,
    movie_name_class="field--name-field-award-film",
) -> dict:
    winner_nominnes_soup = oscar_soup.find_all("div", class_=movie_name_class)
    if winner_nominnes_soup is None:
        return None

    # The first element is the winner, the rest are the nominees
    winner = winner_nominnes_soup[0].text.lower()
    nominees = [nominee.text.lower() for nominee in winner_nominnes_soup[1:]]

    return {"winner": winner, "nominees": nominees}


def parse_movie_winner(
    oscar_soup: BeautifulSoup,
    winner_movie_class="field--name-field-award-film",
) -> str:
    winner_movie_soup = oscar_soup.find("div", class_=winner_movie_class)

    winner_raw = (
        winner_movie_soup.text.lower() if winner_movie_soup is not None else None
    )
    if winner_raw is None:
        return None

    # Remove new lines
    winner = winner_raw.replace("\n", "")
    return winner


def parse_oscar_category(
    oscar_soup: BeautifulSoup,
    category_class="field--name-field-award-category-oscars",
) -> str:
    category_soup = oscar_soup.find("div", class_=category_class)
    return category_soup.text.lower()


# For some categories, the oscars page does not follow the same structure
# as the other categories, so we need to handle them separately
SPECIAL_CATEGORIES = ["international feature film"]


def parse_movie_winner_nominnes_special_category(
    oscar_soup: BeautifulSoup,
    category: str,
) -> str:
    if category == SPECIAL_CATEGORIES[0]:
        # Here the usual winner field contains the corresponding country
        # of the movie, and the winner is in the entity field (it is reversed)
        return parse_movie_winner_nominees(
            oscar_soup, movie_name_class="field--name-field-award-entities"
        )

    else:
        raise ValueError(f"Category {category} is not a special category")


def parse_movie_winner_special_category(
    oscar_soup: BeautifulSoup,
    category: str,
) -> str:
    if category == SPECIAL_CATEGORIES[0]:
        # Here the usual winner field contains the corresponding country
        # of the movie, and the winner is in the entity field (it is reversed)
        return parse_movie_winner(
            oscar_soup, winner_movie_class="field--name-field-award-entities"
        )

    else:
        raise ValueError(f"Category {category} is not a special category")


"""
  Get the oscar winners from the oscars page

  Args:
    page_source: str: The page source of the oscars page
    oscar_categories: list: The oscar categories to scrape, if None, scrape all
"""


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
        if category in SPECIAL_CATEGORIES:
            winner = parse_movie_winner_special_category(oscar_soup, category)
        else:
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
