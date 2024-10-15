"""db operation script"""

import os
import logging
from supabase import create_client

key = os.environ.get("SUPABASE_ANON_KEY")
url = os.environ.get("SUPABASE_URL")
supabase_client = create_client(url, key)


def upsert_data(product, table_name):
    """upsert data to supabase"""
    try:
        _ = (
            supabase_client.table(table_name)
            .upsert(product, on_conflict="product_id")
            .execute()
        )
    except Exception as e:
        logging.error("Database operation failed: %s", e)
        raise


def update_app_setup(shop_url, status):
    """update app setup status in supabase"""
    try:
        _ = (
            supabase_client.table("AppSetup")
            .update({"productSyncStatus": status})
            .eq("shop", shop_url)
            .execute()
        )
    except Exception as e:
        logging.error("Database operation failed: %s", e)
        raise

def product_exists(product_id: str) -> bool:
    """Check if a product exists in Supabase by product_id."""
    try:
        response = (
            supabase_client.table("products")
            .select("product_id")
            .eq("product_id", product_id)
            .limit(1)
            .execute()
        )
        return len(response.data) > 0

    except Exception as e:
        logging.error("Database operation failed: %s", e)
        raise


def upsert_variants(product_id: str, variant_data: list):
    """Insert product variants into the 'variants' table."""
    try:
        _ = (
            supabase_client.table("variants")
            .upsert(variant_data, on_conflict=["variant_id"])
            .execute()
        )

        print(f"Variants for product {product_id} inserted successfully")

    except Exception as e:
        logging.error("Database operation failed: %s", e)
        raise

def create_product(shop, product, text_embeddings, item_type, variant_data):
    """Create a product in Supabase."""
    try:
        content = {
            "title": product["title"],
            "description": product["description"],
            "onlineStoreUrl": product["onlineStoreUrl"],
            "featureImage": product["featuredImage"]["url"],
            "priceRange": product["priceRange"],
        }

        # Insert product into 'products' table
        _= (
            supabase_client.table("products")
            .insert(
                {
                    "shop": shop,
                    "product_id": product["id"],
                    "content": content,
                    "description_embedding": text_embeddings,
                    "product_type": item_type.lower(),
                }
            )
            .execute()
        )

        print(f"Product {product['id']} inserted successfully")

        if variant_data:
            upsert_variants(product["id"], variant_data)

    except Exception as e:
        logging.error("Database operation failed: %s", e)
        raise


def update_product(shop, product, text_embeddings, item_type, variant_data):
    """Update product in Supabase"""
    try:
        content = {
            "title": product["title"],
            "description": product["description"],
            "onlineStoreUrl": product["onlineStoreUrl"],
            "featureImage": product["featuredImage"]["url"],
            "priceRange": product["priceRange"],
        }
        print(content)
        variant_data = variant_data if variant_data else []

        # Prepare the update data
        update_data = {
            "shop": shop,
            "content": content,
            "description_embedding": text_embeddings,
            "product_type": item_type.lower(),
        }

        # Execute the update
        _ = (
            supabase_client.table("products")
            .update(update_data)
            .eq("product_id", product["id"])
            .execute()
        )

        print(f"Product {product['id']} updated successfully")

        # Handle variants update (if required)
        if variant_data:
            upsert_variants(product["id"], variant_data)

    except Exception as e:
        logging.error("Database operation failed: %s", e)
        raise
