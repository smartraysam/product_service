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


def create_update_product(shop, product, text_embeddings, item_type):
    """create/update product in supabase"""
    try:
        response = (
            supabase_client.table("products")
            .upsert(
                {
                    "shop": shop,
                    "product_id": product["id"],
                    "content": {
                        "title": product["title"],
                        "description": product["description"],
                        "onlineStoreUrl": product["onlineStoreUrl"],
                        "featureImage": product["featuredImage"]["url"],
                        "priceRange": product["priceRange"],
                    },
                    "description_embedding": text_embeddings,
                    "product_type": item_type.lower(),
                },
                on_conflict=["product_id"],  # Use 'product_id' as the conflict key
            )
            .execute()
        )

        if response.error:
            logging.error("Failed to insert product: %s", response.error)
        else:
            print(f"Product {product['id']} inserted successfully")

    except Exception as e:
        logging.error("Database operation failed: %s", e)
        raise


def create_update_variant(product, variant, image_embedding):
    """create/update variant in supabase"""
    try:
        response = (
            supabase_client.table("variants")
            .upsert(
                {
                    "product_id": product["id"],
                    "variant_id": variant["node"]["id"],
                    "content": variant["node"],
                    "image_embedding": image_embedding,
                },
                on_conflict=["variant_id"],  # Conflict detection on 'variant_id'
            )
            .execute()
        )
        if response.error:
            logging.error("Failed to insert product: %s", response.error)
        else:
            print(f"Product variant {variant['node']['id']} inserted successfully")

    except Exception as e:
        logging.error("Database operation failed: %s", e)
        raise
