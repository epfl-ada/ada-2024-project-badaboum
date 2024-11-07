import pandas as pd
from bs4 import BeautifulSoup
from argparse import ArgumentParser
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm
import time
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = ArgumentParser(description="Scrape IMDb Reviews")

    parser.add_argument(
        "--output_directory",
        default="imdb_reviews/",
        type=str,
        help="Output file to save the scraped reviews",
    )

    parser.add_argument(
        "--number_years_from_release",
        default=None,
        type=int,
        help="Stop loading reviews after this number of years from the release date (reviews beyond this date might still be scrapped), if None, all reviews are loaded and scraped",
    )

    parser.add_argument(
        "--input_dataset_path",
        required=True,
        type=str,
        help="Path to the input dataset",
    )

    parser.add_argument(
        "--imdb_id_column",
        default="imdb_id",
        type=str,
        help="Name of the column containing the IMDb ID",
    )

    parser.add_argument(
        "--release_year_column",
        default="release_year",
        type=str,
        help="Name of the column containing the release year",
    )

    args = parser.parse_args()
    return args


def get_url(imdb_id: str) -> str:
    BASE_URL = "https://www.imdb.com/title/"
    ASC_SORT_NO_SPOILERS = (
        "/reviews/?ref_=tt_ov_urv&spoilers=EXCLUDE&sort=submission_date%2Casc"
    )

    return BASE_URL + imdb_id + ASC_SORT_NO_SPOILERS


def parse_review(review_soup: BeautifulSoup):
    review_text_raw = review_soup.find("div", class_="ipc-html-content-inner-div")
    review_text = review_text_raw.get_text(strip=True) if review_text_raw else None

    rating_raw = review_soup.find("span", class_="ipc-rating-star--rating")
    rating = rating_raw.get_text(strip=True) if rating_raw else None

    review_title_tag = review_soup.find("h3", class_="ipc-title__text")
    review_title = review_title_tag.text if review_title_tag else None

    review_date_raw = review_soup.find("li", class_="review-date")
    review_date_str = review_date_raw.get_text(strip=True) if review_date_raw else None
    review_date = (
        datetime.strptime(review_date_str, "%b %d, %Y") if review_date_str else None
    )

    return review_title, review_text, rating, review_date


def get_reviews_soup_list(page_source: str) -> list:
    soup = BeautifulSoup(page_source, "html.parser")
    return soup.find_all("article", class_="user-review-item")


def get_most_recent_review_year(page_source: str) -> int:
    review_soups = get_reviews_soup_list(page_source)

    # Reviews are scrapped in ascending order of date
    # Thus, the most recent review is the last one
    _, _, _, most_recent_review_date = parse_review(review_soups[-1])
    return most_recent_review_date.year


def scrape_reviews(page_source: str):
    review_soups = get_reviews_soup_list(page_source)

    reviews_data = []
    for review_soup in review_soups:
        review_title, review_text, rating, review_date = parse_review(review_soup)
        reviews_data.append(
            {
                "title": review_title,
                "text": review_text,
                "rating": rating,
                "date": review_date,
            }
        )

    return reviews_data


def get_nb_loaded_reviews(page_source: str) -> int:
    soup = BeautifulSoup(page_source, "html.parser")
    review_soups = soup.find_all("article", class_="user-review-item")
    return len(review_soups)


def create_driver() -> webdriver.Firefox:
    options = webdriver.FirefoxOptions()
    options.add_argument("--headless")
    driver = webdriver.Firefox(options=options)
    return driver


def load_all_reviews(
    driver: webdriver.Firefox,
    max_year: int = None,
    time_between_click_trials: int = 1,
    time_between_review_loading: int = 2,
    max_click_trials: int = 3,
    timeout: int = 10,
):
    span_container = WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CLASS_NAME, "chained-see-more-button"))
    )

    # Find the button within this span
    see_more_button = span_container.find_element(By.CLASS_NAME, "ipc-see-more__button")

    # Scroll to the button using JavaScript
    driver.execute_script("arguments[0].scrollIntoView(true);", see_more_button)

    # Try to click the button
    click_trials = 0
    click_success = False
    while not click_success:
        try:
            time.sleep(time_between_click_trials)
            see_more_button.click()
            click_success = True
        except Exception as e:
            if click_trials < max_click_trials:
                click_trials += 1
            else:
                raise

    # Wait for all reviews to be loaded
    old_nb_reviews = -1
    current_nb_reviews = get_nb_loaded_reviews(driver.page_source)

    most_recent_review_year = get_most_recent_review_year(driver.page_source)

    # Load reviews until the number of loaded reviews does not change
    # or until the most recent review is older than the stop year
    while current_nb_reviews != old_nb_reviews and (
        max_year == None or most_recent_review_year <= max_year
    ):
        old_nb_reviews = current_nb_reviews
        time.sleep(time_between_review_loading)
        current_nb_reviews = get_nb_loaded_reviews(driver.page_source)
        most_recent_review_year = get_most_recent_review_year(driver.page_source)
        print(f"Reviews loaded for current movie: {current_nb_reviews}", end="\r")


def scrape_all_reviews(
    driver: webdriver.Firefox, base_url: str, max_year: int = None
) -> list:
    driver.get(base_url)

    # Load all reviews
    try:
        load_all_reviews(driver, max_year=max_year)
    except Exception as e:
        logging.error(f"Error loading reviews for {base_url}")
        raise

    # Scrape reviews
    reviews = scrape_reviews(driver.page_source)

    return reviews


def main():
    args = parse_args()
    driver = create_driver()

    # Load the dataset
    dataset = pd.read_csv(args.input_dataset_path)

    # Create the output directory
    os.makedirs(args.output_directory, exist_ok=True)

    # Scraping reviews for each movie
    for _, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        imdb_id = row[args.imdb_id_column]
        release_year = row[args.release_year_column]

        # Get the URL
        url = get_url(imdb_id)

        # Scrape the reviews
        try:
            reviews_data = scrape_all_reviews(
                driver, url, max_year=release_year + args.number_years_from_release
            )
        except Exception as e:
            logging.error(f"Error scraping reviews for {imdb_id}")
            continue

        # Create a DataFrame
        reviews_df = pd.DataFrame(reviews_data)

        # Add the IMDb ID
        reviews_df["imdb_id"] = imdb_id

        # Save the data
        reviews_df.to_csv(f"{args.output_directory}/{imdb_id}_reviews.csv", index=False)
        logging.info(
            f"Scraped {len(reviews_data)} reviews for {imdb_id}. Data saved to {imdb_id}_reviews.csv"
        )

    driver.quit()


if __name__ == "__main__":
    main()
