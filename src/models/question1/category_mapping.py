# Description: This file contains the mapping of the original categories to the new categories.

# Define the new categories (broader categories)
new_categories = [
    "acting",
    "directing and production",
    "writing",
    "cinematography",
    "art direction",
    "sound",
    "music",
    "editing and effects",
    "costume",
]

# Define the mapping from the original categories to the new categories
# Note : The mapping was generated with the help of ChatGPT 4o model
# The following prompt was used: "Given the original categories and the new ones below,
# could you define a python dictionay that maps the original categories to the new ones?"
category_mapping = {
    "actor in a supporting role": "acting",
    "actress in a supporting role": "acting",
    "actress": "acting",
    "actor": "acting",
    "actress in a leading role": "acting",
    "actor in a leading role": "acting",
    "directing": "directing and production",
    "best picture": "directing and production",
    "outstanding production": "directing and production",
    "outstanding motion picture": "directing and production",
    "assistant director": "directing and production",
    "dance direction": "directing and production",
    "writing (screenplay written directly for the screen)": "writing",
    "writing (screenplay)": "writing",
    "writing (screenplay—based on material from another medium)": "writing",
    "writing (original screenplay)": "writing",
    "writing (adapted screenplay)": "writing",
    "writing (screenplay based on material previously produced or published)": "writing",
    "writing (original story)": "writing",
    "writing (motion picture story)": "writing",
    "writing (story and screenplay)": "writing",
    "writing (original motion picture story)": "writing",
    "writing (story and screenplay—written directly for the screen)": "writing",
    "writing (story and screenplay—based on factual material or material not previously published or produced)": "writing",
    "cinematography": "cinematography",
    "cinematography (black-and-white)": "cinematography",
    "cinematography (color)": "cinematography",
    "art direction": "art direction",
    "art direction (black-and-white)": "art direction",
    "art direction (color)": "art direction",
    "sound": "sound",
    "sound recording": "sound",
    "sound effects editing": "sound",
    "sound editing": "sound",
    "sound mixing": "sound",
    "music (original score)": "music",
    "music (song)": "music",
    "music (original song)": "music",
    "music (music score of a dramatic or comedy picture)": "music",
    "music (scoring of a musical picture)": "music",
    "music (scoring)": "music",
    "music (original dramatic score)": "music",
    "music (original musical or comedy score)": "music",
    "music (music score—substantially original)": "music",
    "music (scoring of music—adaptation or treatment)": "music",
    "music (song—original for the picture)": "music",
    "film editing": "editing and effects",
    "visual effects": "editing and effects",
    "special effects": "editing and effects",
    "makeup": "editing and effects",
    "costume design": "costume",
    "costume design (black-and-white)": "costume",
    "costume design (color)": "costume",
}


import pandas as pd

"""
  Map the original categories to the new categories.
  Drop entries that do not have a mapping.
"""


def map_categories(oscar_movies_df: pd.DataFrame):
    oscar_movies_df["oscar_category"] = oscar_movies_df["oscar_category"].map(
        category_mapping
    )

    # Drop entries that do not have a mapping
    oscar_movies_df = oscar_movies_df.dropna(subset=["oscar_category"])

    return oscar_movies_df
