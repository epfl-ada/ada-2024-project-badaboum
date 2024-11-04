import pandas as pd
from bs4 import BeautifulSoup
from argparse import ArgumentParser
import logging
from selenium import webdriver
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = ArgumentParser(description="Scrape IMDb Reviews")

    parser.add_argument(
        "--base_url",
        default="https://www.imdb.com/title/tt15398776/reviews/?ref_=tt_ov_urv",
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
        "--max_reviews",
        default=2,
        type=int,
        help="Maximum number of reviews to scrape",
    )

    args = parser.parse_args()
    return args

def parse_review(
        review_soup: BeautifulSoup
):
    review_text = review_soup.find("div", class_="text show-more__control").get_text(strip=True)
    rating_tag = review_soup.find("span", class_="rating-other-user-rating")
    rating = rating_tag.find("span").text if rating_tag else None
    review_title = review_soup.find("a", class_="title").get_text(strip=True)
    
    # Extracting the date of the review
    date_tag = review_soup.find("span", class_="review-date")
    review_date = date_tag.text if date_tag else None

    return review_title, review_text, rating, review_date

def scrape_reviews(
        page_source: str, 
        max_reviews: int
):
    soup = BeautifulSoup(page_source, "html.parser")
    review_soups = soup.find_all("div", class_="review-container")

    reviews_data = []
    for review_soup in review_soups:
        if len(reviews_data) >= max_reviews:
            break
        review_title, review_text, rating, review_date = parse_review(review_soup)
        reviews_data.append({
            "title": review_title,
            "text": review_text,
            "rating": rating,
            "date": review_date
        })

    return reviews_data

def create_driver() -> webdriver.Firefox:
    options = webdriver.FirefoxOptions()
    options.add_argument("--headless")
    driver = webdriver.Firefox(options=options)
    return driver

def get_page_source(
        driver: webdriver.Firefox, 
        base_url: str, 
        max_reviews: int
    ) -> list:
    driver.get(base_url)

    reviews_data = []
    
    while len(reviews_data) < max_reviews:

        # Fetch the current page source
        page_source = driver.page_source
        new_reviews = scrape_reviews(page_source, max_reviews - len(reviews_data))
        
        if new_reviews:
            reviews_data.extend(new_reviews)

        logging.info(f"Scraped {len(reviews_data)} reviews so far...")

    return reviews_data[:max_reviews]  # Return only the requested number of reviews

def main():
    args = parse_args()
    driver = create_driver()

    try:
        # Get the reviews data and scrape reviews
        reviews_data = get_page_source(driver, args.base_url, args.max_reviews)

        # Create a DataFrame
        reviews_df = pd.DataFrame(reviews_data)
        
        # Save the data
        reviews_df.to_csv(args.output_file, index=False)
        logging.info(f"Scraped {len(reviews_data)} reviews. Data saved to {args.output_file}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
