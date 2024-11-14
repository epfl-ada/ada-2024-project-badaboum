
# Oscar or Flopscar: Do the Academy Awards Reflect Audience Taste?

## Quickstart

```bash
# clone project
# via HTTPS
git clone https://github.com/epfl-ada/ada-2024-project-badaboum.git

# OR via SSH
git clone git@github.com:epfl-ada/ada-2024-project-badaboum.git
cd ada-2024-project-badaboum

# [OPTIONAL] create conda environment
conda create -n <env_name> python=3.11
conda activate <env_name>


# install requirements
pip install -r pip_requirements.txt
```



### How to use the library

#### Downloading the Data
You have two options for obtaining the datasets:

**Option A**: Google Drive Download
Download the complete dataset folder on [Google Drive](https://drive.google.com/drive/folders/15Ug1HI5YHSo6eIUCWqpsr4PWtREzisau?usp=sharing) here and place it in the data directory.

**Option B**: Running Data Download Scripts
Alternatively, you can  set up the datasets yourself.
1. Download base datasets from the following links:
    - [CMU Dataset](https://www.cs.cmu.edu/~ark/personas/)
    - [IMDb Dataset](https://datasets.imdbws.com/)
    - [Golden Globes](https://www.kaggle.com/datasets/unanimad/golden-globe-awards)
    - [BAFTA Awards](https://www.kaggle.com/datasets/unanimad/bafta-awards)
2. Place the datasets in the ```data``` directory.
3. Run the following scripts to create the remaining data:
    ```bash
    ```

Your data directory should look like this:
```
├── imdb                  <- IMDb dataset
|       ├── name.basics.tsv
|       ├── title.basics.tsv
|       ├── title.crew.tsv
|       ├── title.rating.tsv
├── imdb_reviews               <- IMDb reviews dataset
|       ├── imdb_reviews_best_picture_2years_from_release.csv
|       ├── imdb_reviews_with_compound.csv
├── MoviesSummaries       <- CMU dataset
│       ├── character.metadata.tsv
│       ├── movie.metadata.tsv
│       ├── name.clusters.txt
│       ├── plot_summaries.txt
│       ├── README.txt
│       ├── tvtropes.clusters.txt
├── other_awards          <- Other awards dataset
|       ├── bafta_films.csv
|       ├── golden_globe_awards.csv
|       ├── other_awards.csv
├── all_other_movies.csv <- All movies that are not Oscar winners nor nominees
├── oscar_movies.csv     <- Oscar winners, merged with IMDb and CMU data
├── oscar_winners_1929_2016.csv     <- Oscar winners, scrapped from the Oscars website
```


## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
││
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```

## Project Proposal

### Abstract
This project explores the relationship between Oscar-winning films and audience ratings to assess the alignment (or divergence) between the Academy’s choices and public opinion. We will analyze IMDb ratings of Oscar winners, non-winning nominees, and box-office hits to uncover any trends or biases in Academy Awards decisions over time. Additionally, we aim to examine the influence of star power, genre, and the "Oscar bump" effect on ratings, as well as compare Oscars to other international film festivals. Through visualizations and statistical measures, this research will reveal patterns in Oscar nominations and wins, contributing to discussions on cultural and rating biases.

### Research Questions
- Do Oscar-winning films generally have higher ratings compared to non-winning nominees and top box-office hits?
- How do ratings of Oscar-winning films evolve over time—do they "age well" with audiences?
- Is there a correlation between high-profile actors/directors and Oscar wins, regardless of ratings?
- How does the Oscars’ preference align (or diverge) from that of other major international film festivals?
- Does winning an Oscar lead to a measurable increase in ratings or review counts (i.e., the "Oscar bump")?
- Are there discernible biases within Oscar winners, such as genre, nationality, or star power, that differ from audience preferences?


### Methods
Data collection:
- Downloading CMU, IMDB and Kaggle datasets
- Web scraping IMDb for ratings, reviews, and other metadata.
- Querying the Oscars database for nominees and winners.

Data processing:
- Merging IMDB and CMU datasets to create a unified database.

Data analysis:
- Sentiment analysis of reviews to assess audience reception.
- Time series analysis of ratings to track changes over time.
- Statistical tests to compare ratings between Oscar winners, nominees, and box-office hits.
- Visualization of trends and biases in Oscar nominations and wins.

### Proposed timeline
#### Milestone 2 (deadline Nov 15)
Week 1 (Oct 28 - Nov 4):
- Define Research Questions: Finalize research questions based on individual ideas from P1.
- Data Pipeline Setup: Start building the data pipelines, focusing on web scraping for Oscar winners and IMDb query functionality.
- README Draft: Begin drafting the README, outlining the proposal, abstract, and research questions.

Week 2 (Nov 4 - Nov 11):
- Initial Data Collection and Cleaning: Start collecting data from IMDb, CMU, and other sources. Clean and organize data for analysis.
- Prototype Development: Create basic prototypes for key analyses.
- Code Documentation.

Week 3 (Nov 11 - Nov 18):
- Notebook Preparation: Prepare an initial Jupyter notebook showing initial results, with basic descriptive statistics and pipeline verification.
- Finalize README.
- Submit Milestone 2.

#### Milestone 3 (deadline Dec 20)
Week 4 (Nov 18 - Nov 25):
- Detailed Analysis Implementation: Continue and refine key analyses based on initial results, focusing on advanced techniques and statistical tests.
- Data Story Outline: Create a structured outline of the data story, planning the story and highlighting key insights.

Week 5 (Nov 25 - Dec 2):
- Finalize Analysis: Complete all analyses and visualizations.
- Notebook Refinement: Update the notebook with final analyses, documenting findings.
- Data Story Drafting: Begin writing the data story, integrating key visuals and findings.

Week 6 (Dec 2 - Dec 9):
- Finalize Data Story.
- Code Optimization and Documentation.

Week 7 (Dec 9 - Dec 16):
- Final Review: Polish the data story, README, and notebook. Final team review
- Submit Milestone 3


### Organization within the team
Each member was responsible for at least one research question. The other members would provide feedback and suggestions to improve the analysis. The team would meet weekly to discuss progress, challenges, and next steps. The team would also share resources and insights to support each other’s work.
The other tasks were distributed as evenly as possible, based on the workload of the assigned research questions.
