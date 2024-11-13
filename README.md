
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
[TODO: Tell us how the code is arranged, any explanations goes here.]



## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
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

### Datasets
- [CMU Dataset](https://www.cs.cmu.edu/~ark/personas/)
- IMDb Dataset
- IMDB reviews
- Official Oscars Website
- Golden globes and BAFTA awards (Kaggle)

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
- Prototype Development: Create basic prototypes for key analyses, such as the Rating Gap Index and bias mapping.
- Code Documentation.
- Notebook Preparation: Prepare an initial Jupyter notebook showing initial results, with basic descriptive statistics and pipeline verification.
- Finalize README.
- Submit Milestone 2.

#### Milestone 3 (deadline Dec 20)
Week 3 (Nov 11 - Nov 18):
- Advanced Analysis Development: Begin implementing more detailed analyses, focusing on the Rating Gap Index, the “Oscar bump” effect, and any other comparisons.
- Visualization Prototypes: Start creating initial visualizations, such as timelines and heatmaps, to showcase early findings.

Week 4 (Nov 18 - Nov 25):
- Complete Core Analyses: Finalize core analyses, ensuring each research question has a concrete and reliable approach.
- Finalize Visualizations.
- Data Story Planning: Outline the data story structure, identifying key findings to highlight.

Week 5 (Nov 25 - Dec 2):
- Data Story Drafting: Begin writing the data story, integrating key visuals and findings.
- Notebook Refinement: Update the notebook with final analyses, ensuring clear and comprehensive documentation of each step.

Week 6 (Dec 2 - Dec 9):
- Finalize Data Story.
- Code Optimization and Documentation.

Week 7 (Dec 9 - Dec 16):
- Final Review: Polish the data story, README, and notebook. Final team review
- Submit Milestone 3


### Organization within the team
Each member was responsible for at least one research question. The other members would provide feedback and suggestions to improve the analysis. The team would meet weekly to discuss progress, challenges, and next steps. The team would also share resources and insights to support each other’s work.
The other tasks were distributed as evenly as possible, based on the workload of the assigned research questions.
