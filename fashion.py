import aiohttp
from fashion_clip.fashion_clip import FashionCLIP
from PIL import Image
import io
from supabase import create_client, Client
import requests
import numpy as np
from scipy.spatial.distance import cosine
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio
import urllib.parse
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

key = os.environ.get("SUPABASE_ANON_KEY")
url = os.environ.get("SUPABASE_URL")

model = FashionCLIP("fashion-clip")
supabase: Client = create_client(url, key)

async def fetch_embeddings_async(shop, item_type):
    # Make asynchronous HTTP requests using aiohttp, or async database queries
    async with aiohttp.ClientSession() as session:
        async with session.get(f'https://ugbtvxiaydwqsjspaghc.supabase.co/rest/v1/products?select=%2A%2C%20variants%28%2A%29&shop=eq.{urllib.parse.quote_plus(shop)}&product_type=eq.{item_type}&apikey={urllib.parse.quote_plus(key)}') as response:
            return await response.json()

def generate_embedding(user_input):
    # This function should run your model's inference (synchronously)
    start_embedding = time.time()
    user_embedding = model.encode_text([user_input], batch_size=5)
    end_embedding = time.time()
    print(f"encode text ran in {round(end_embedding - start_embedding, 2)}s")

    if user_embedding.ndim > 1:
        user_embedding = user_embedding.flatten()

    return user_embedding

async def recommend_outfits_with_embeddings(user_embedding, shop, item_type):
    loop = asyncio.get_running_loop()

    with ThreadPoolExecutor() as pool:
        embeddings = await fetch_embeddings_async(shop, item_type)
        print(len(embeddings))

    # Calculate cosine similarity between user input and each product embedding
    recommendations = []
    for product in embeddings:
        product_id = product["product_id"]
        text_embedding = np.array(product["description_embedding"])
        text_similarity = 1 - cosine(user_embedding, text_embedding)

        for variant in product["variants"]:
            # Calculate cosine similarity for both image and text embeddings
            image_embedding = np.array(variant['image_embedding'])
            image_similarity = 1 - cosine(user_embedding, image_embedding)
            
            # Aggregate similarity (you can use an average or weighted sum)
            aggregated_similarity = (image_similarity + text_similarity) / 2

            recommendations.append(
                {
                    "product_id": product_id,
                    "variant_id": variant["variant_id"],
                    "similarity": aggregated_similarity,
                    "product_content": product["content"],  # Full product details
                    "item_type": product["product_type"],
                    "variants": product["variants"],
                }
            )

    # Sort recommendations by similarity (descending order)
    recommendations = sorted(recommendations, key=lambda x: x['similarity'], reverse=True)
    return recommendations[0]


def get_image_from_url(url):
    response = requests.get(url)
    try:
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Error retrieving image: {e}")
    return image


def embed_text(description):
    # Generate text embedding
    text_embedding = model.encode_text([description], 1).flatten().tolist()
    return text_embedding


def embed_image(image_url):
    image = get_image_from_url(image_url)
    # Generate image embedding
    image_embedding = model.encode_images([image], 1).flatten().tolist()
    return image_embedding
