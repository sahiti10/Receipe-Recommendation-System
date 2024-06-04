import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle 
import config

# Load in parsed recipe dataset
df_recipes = pd.read_csv(config.PARSED_PATH)
df_recipes['ingredients_parsed'] = df_recipes.ingredients_parsed.values.astype('U')

# Combine ingredients, spice level, and cuisine type into a single text column for vectorization
df_recipes['combined_features'] = df_recipes['ingredients_parsed'] + " " + df_recipes['spice_level'] + " " + (df_recipes['cuisine_type'] if 'cuisine_type' in df_recipes.columns else '')

# TF-IDF feature extractor on combined features
tfidf = TfidfVectorizer()
tfidf.fit(df_recipes['combined_features'])
tfidf_recipe = tfidf.transform(df_recipes['combined_features'])

# Save the tfidf model and encodings 
with open(config.TFIDF_MODEL_PATH, "wb") as f:
    pickle.dump(tfidf, f)

with open(config.TFIDF_ENCODING_PATH, "wb") as f:
    pickle.dump(tfidf_recipe, f)

# Save the updated dataset with reviews
df_recipes.to_csv(config.PARSED_PATH, index=False)


