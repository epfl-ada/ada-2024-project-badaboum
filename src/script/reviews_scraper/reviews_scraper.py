import pandas as pd
from bs4 import BeautifulSoup
from argparse import ArgumentParser
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = ArgumentParser(description="Scrape IMDb Reviews")

    parser.add_argument(
        "--base_url",
        default="https://www.imdb.com/title/tt15398776/reviews/?ref_=tt_ov_urv&spoilers=EXCLUDE&sort=submission_date%2Casc",
        type=str,
        help="Base URL to the IMDb reviews page",
    )

    parser.add_argument(
        "--output_file",
        default="imdb_reviews.csv",
        type=str,
        help="Output file to save the scraped reviews",
    )

    parser.add_argument(
        "--stop_year_from_release",
        default=None,
        type=int,
        help="Stop scraping reviews after this number of years from the release date, if None, all reviews are scraped",
    )

    parser.add_argument(
        "--input_dataset_name",
        required=True,
        type=str,
        help="Name of the input dataset",
    )

    parser.add_argument(
        "--imdb_id_column",
        default="imdb_id",
        type=str,
        help="Name of the column containing the IMDb ID",
    )

    parser.add_argument(
        "--release_year_column",
        default="release_date",
        type=str,
        help="Name of the column containing the release year",
    )

    parser.add_argument(
        "--wait_time_to_scroll_down",
        default=2,
        type=int,
        help="Time to when scrolling down the page (in seconds)",
    )

    parser.add_argument(
        "--wait_time_between_reviews_load",
        default=2,
        type=int,
        help="Time to wait between reviews load (in seconds)",
    )

    args = parser.parse_args()
    return args


def parse_review(review_soup: BeautifulSoup):
    review_text_raw = review_soup.find("div", class_="ipc-html-content-inner-div")
    review_text = review_text_raw.get_text(strip=True) if review_text_raw else None

    rating_raw = review_soup.find("span", class_="ipc-rating-star--rating")
    rating = rating_raw.get_text(strip=True) if rating_raw else None

    review_title_tag = review_soup.find("h3", class_="ipc-title__text")
    review_title = review_title_tag.text if review_title_tag else None

    review_date_raw = review_soup.find("li", class_="review-date")
    review_date = review_date_raw.get_text(strip=True) if review_date_raw else None

    return review_title, review_text, rating, review_date


def scrape_reviews(page_source: str, max_reviews: int):
    soup = BeautifulSoup(page_source, "html.parser")
    review_soups = soup.find_all("article", class_="user-review-item")

    reviews_data = []
    for review_soup in review_soups:
        if len(reviews_data) >= max_reviews:
            break
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


def scrape_all_reviews(
    driver: webdriver.Firefox, base_url: str, stop_year: int = None
) -> list:
    driver.get(base_url)

    # Perform a click on the button to see all reviews
    try:
        # Locate the span with class "chained-see-more-button"
        span_container = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "chained-see-more-button"))
        )

        # Find the button within this span
        see_more_button = span_container.find_element(
            By.CLASS_NAME, "ipc-see-more__button"
        )

        # Scroll to the button using JavaScript
        driver.execute_script("arguments[0].scrollIntoView(true);", see_more_button)

        time.sleep(4)

        see_more_button.click()

        # Wait for all reviews to be loaded
        old_nb_reviews = -1
        current_nb_reviews = get_nb_loaded_reviews(driver.page_source)

        while current_nb_reviews != old_nb_reviews:
            old_nb_reviews = current_nb_reviews
            time.sleep(2)
            current_nb_reviews = get_nb_loaded_reviews(driver.page_source)
            logging.info(f"Loaded {current_nb_reviews} reviews so far...")

    except Exception as e:
        print(f"An error occurred while trying to click the button: {e}")

    reviews = scrape_reviews(driver.page_source, max_reviews=1000)
    logging.info(f"Scraped {len(reviews)} reviews so far...")

    return reviews


def main():
    args = parse_args()
    driver = create_driver()

    try:
        # Get the reviews data and scrape reviews
        # reviews_data = get_page_source(driver, args.base_url, args.max_reviews)
        reviews_data = scrape_all_reviews(driver, args.base_url)

        # Create a DataFrame
        reviews_df = pd.DataFrame(reviews_data)

        # Save the data
        reviews_df.to_csv(args.output_file, index=False)
        logging.info(
            f"Scraped {len(reviews_data)} reviews. Data saved to {args.output_file}"
        )
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
