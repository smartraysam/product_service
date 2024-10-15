"""process product script."""

import json
import os
from supabase import create_client, Client
from openai import OpenAI
from db import update_app_setup, create_product, product_exists, update_product
from fashion import embed_text, embed_image

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY")
)

# Initialize OpenAI client
openai = OpenAI(api_key=os.getenv("OPEN_API_KEY"))


def fetch_product_category(product):
    """Fetch product category using OpenAI GPT."""
    try:
        response = openai.chat.completions.create(
            response_format={"type": "json_object"},
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Categorize a product given the title and description.\n"
                        "The response should be a JSON object with a single field: productCategory (string).\n"
                        f"Product info: {product['title']} - {product['description']}"
                    ),
                }
            ],
        )
        response_dict = response.model_dump()
        return json.loads(
            response_dict["choices"][0]["message"]["content"] or "{}"
        ).get("productCategory")
    except Exception as e:
        print(f"Error fetching product category for {product['title']}: {e}")
        return None


def process_variant(variant, product, image_embedding_cache):
    """Process a product variant, including embedding images."""
    image_url = (
        variant["node"]["image"]["url"]
        if variant["node"].get("image") is not None
        else product["featuredImage"]["url"]
    )
    image_embedding = image_embedding_cache.get(image_url)
    print("product Image URL:", image_url)
    if not image_embedding:
        try:
            image_embedding = embed_image(image_url)
            image_embedding_cache[image_url] = image_embedding
        except Exception as e:
            print(f"Error embedding image for {image_url}: {e}")
            image_embedding = []

    return {
        "product_id": product["id"],
        "variant_id": variant["node"]["id"],
        "content": variant["node"],
        "image_embedding": image_embedding,
    }


def handle_product_sync(products, shop):
    """Sync products from Shopify to Supabase and compute embeddings."""
    image_embedding_cache = {}
    for product in products:
        item_type = fetch_product_category(product)
        print(item_type)
        variant_data = []
        for variant in product["variants"]["edges"]:
            variant_data.append(
                process_variant(variant, product, image_embedding_cache)
            )

        descriptions = f"{product['title']} - {product['description']}"
        text_embeddings = embed_text(descriptions)
        if not product_exists(product["id"]):
            create_product(shop, product, text_embeddings, item_type, variant_data)
        else:
            update_product(shop, product, text_embeddings, item_type, variant_data)

    update_app_setup(shop, "COMPLETED")
    return {"status": "success"}


def handle_product_update(product, shop):
    """Sync products from Shopify to Supabase and compute embeddings."""
    image_embedding_cache = {}

    item_type = fetch_product_category(product)
    print(item_type)
    variant_data = []
    for variant in product["variants"]["edges"]:
        variant_data.append(process_variant(variant, product, image_embedding_cache))

    descriptions = f"{product['title']} - {product['description']}"
    text_embeddings = embed_text(descriptions)

    if product_exists(product["id"]):
        update_product(shop, product, text_embeddings, item_type, variant_data)
    else:
        create_product(shop, product, text_embeddings, item_type, variant_data)
    return {"status": "success"}
