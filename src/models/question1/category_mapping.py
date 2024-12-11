# Description: This file contains the mapping of the original categories to the new categories.

# Define the new categories (broader categories)
new_categories = [
    "direction",
    "writing",
    "best picture",
    "music",
    "sound",
    "acting",
    "cinematography",
    "costume",
    "effects",
    "editing",
]

# Define the mapping from the original categories to the new categories
# This mapping has been done with the application of K-means clustering followed by manual inspection for finetuning
category_mapping = {
    # cluster 0: direction
    "art direction": "direction",
    "art direction (black-and-white)": "direction",
    "art direction (color)": "direction",
    "directing": "direction",
    "assistant director": "direction",
    "dance direction": "direction",
    # cluster 1: writing
    "writing (original story)": "writing",
    "writing (screenplay)": "writing",
    "writing (original screenplay)": "writing",
    "writing (original motion picture story)": "writing",
    "documentary (feature)": "writing",
    "writing (motion picture story)": "writing",
    "writing (story and screenplay)": "writing",
    "writing (screenplay—based on material from another medium)": "writing",
    "writing (story and screenplay—written directly for the screen)": "writing",
    "writing (story and screenplay—based on factual material or material not previously published or produced)": "writing",
    "writing (screenplay based on material from another medium)": "writing",
    "writing (screenplay written directly for the screen)": "writing",
    "writing (screenplay based on material previously produced or published)": "writing",
    "writing (adapted screenplay)": "writing",
    # cluster 2: best picture
    "outstanding production": "best picture",
    "outstanding motion picture": "best picture",
    "best motion picture": "best picture",
    "best picture": "best picture",
    # cluster 3: music
    "music (song)": "music",
    "music (scoring)": "music",
    "music (original score)": "music",
    "music (scoring of a musical picture)": "music",
    "music (music score of a dramatic picture)": "music",
    "music (music score of a dramatic or comedy picture)": "music",
    "music (original song)": "music",
    "music (scoring of music—adaptation or treatment)": "music",
    "music (music score—substantially original)": "music",
    "music (song—original for the picture)": "music",
    "music (original dramatic score)": "music",
    "music (original musical or comedy score)": "music",
    # cluster 4: sound
    "sound recording": "sound",
    "sound": "sound",
    "sound effects editing": "sound",
    "sound editing": "sound",
    "sound mixing": "sound",
    # cluster 5: acting
    "actor": "acting",
    "actress": "acting",
    "actress in a supporting role": "acting",
    "actor in a supporting role": "acting",
    "actress in a leading role": "acting",
    "actor in a leading role": "acting",
    # cluster 6: cinematography
    "cinematography": "cinematography",
    "cinematography (color)": "cinematography",
    "cinematography (black-and-white)": "cinematography",
    # cluster 7: costume
    "makeup": "costume",
    "costume design (color)": "costume",
    "costume design (black-and-white)": "costume",
    "costume design": "costume",
    # cluster 8: effects
    "special effects": "effects",
    "visual effects": "effects",
    # cluster 9: editing
    "film editing": "editing",
    "foreign language film": "editing",
    "animated feature film": "editing",
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
