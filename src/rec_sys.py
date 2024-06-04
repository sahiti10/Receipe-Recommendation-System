import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from ingredient_parser import ingredient_parser
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pickle
import config 
import unidecode, ast

def RecSys(ingredients, spice_level, cuisine_type=None, N=5):
    # Load in the TFIDF model and encodings for combined features
    with open(config.TFIDF_ENCODING_PATH, 'rb') as f:
        tfidf_encodings = pickle.load(f)

    with open(config.TFIDF_MODEL_PATH, "rb") as f:
        tfidf = pickle.load(f)

    # Load df_recipes
    df_recipes = pd.read_csv(config.PARSED_PATH)

    # Parse the ingredients using the ingredient_parser
    try: 
        ingredients_parsed = ingredient_parser(ingredients)
    except:
        ingredients_parsed = ingredient_parser([ingredients])

    # Combine features with optional cuisine type
    combined_features = ingredients_parsed + " " + spice_level
    if cuisine_type:
        combined_features += " " + cuisine_type

    # Encode the input combined features using the TFIDF model
    ingredients_tfidf = tfidf.transform([combined_features])

    # Calculate cosine similarity
    cos_sim = [cosine_similarity(ingredients_tfidf, x).flatten()[0] for x in tfidf_encodings]

    # Sort by cosine similarity score
    top_indices = sorted(range(len(cos_sim)), key=lambda i: cos_sim[i], reverse=True)[:N]

    # Store recommendations
    recommendations = []
    for idx in top_indices:
        recommendation = {
            'recipe': df_recipes.at[idx, 'recipe_name'],
            'ingredients': df_recipes.at[idx, 'ingredients'],
            'cuisine': df_recipes.at[idx, 'cuisine_type'] if 'cuisine_type' in df_recipes.columns else 'Not specified',
            'user_review': df_recipes.at[idx, 'user_review'],
            'url': df_recipes.at[idx, 'recipe_urls']
        }
        recommendations.append(recommendation)

    return recommendations
