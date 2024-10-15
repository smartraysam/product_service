"""process product script."""
import json
import os
from supabase import create_client, Client
from openai import OpenAI
from db import update_app_setup, create_update_product, create_update_variant
from fashion import embed_text, embed_image

# Initialize Supabase client
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


def process_variant(variant, product):
    """Process a product variant, including embedding images."""
    variant_option_map = {}
    image_embedding_cache = {}

    for option in variant["node"]["selectedOptions"]:
        variant_option_map.setdefault(option["name"], []).append(option["value"])

    image_url = (
        variant["node"].get("image", {}).get("url", product["featuredImage"]["url"])
    )
    image_embedding = image_embedding_cache.get(image_url)

    if not image_embedding:
        try:
            image_embedding = embed_image(image_url)
            image_embedding_cache[image_url] = image_embedding
        except Exception as e:
            print(f"Error embedding image for {image_url}: {e}")
            image_embedding = []

    create_update_variant(product, variant, image_embedding)


def handle_product_sync(products, shop):
    """Sync products from Shopify to Supabase and compute embeddings."""
    update_app_setup(shop, "IN_PROGRESS")

    for product in products:
        item_type = fetch_product_category(product)
        print(item_type)

        for variant in product["variants"]["edges"]:
            process_variant(variant, product)

        descriptions = f"{product['title']} - {product['description']}"
        text_embeddings = embed_text(descriptions)
        create_update_product(shop, product, text_embeddings, item_type)
    update_app_setup(shop, "COMPLETED")


def handle_webhook(product, shop):
    """handle webhook event."""
    item_type = fetch_product_category(product)
    print(item_type)

    for variant in product["variants"]["edges"]:
        process_variant(variant, product)

    descriptions = f"{product['title']} - {product['description']}"
    text_embeddings = embed_text(descriptions)
    create_update_product(shop, product, text_embeddings, item_type)
