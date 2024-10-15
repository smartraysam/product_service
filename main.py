"""main script."""

import asyncio
import json
import logging
import os
import requests
from dotenv import load_dotenv
from fashion_clip.fashion_clip import FashionCLIP
from flask import Flask, jsonify, request
from openai import OpenAI
from pydantic import BaseModel, conlist
from supabase.client import create_client
from fashion import embed_image, embed_text, recommend_outfits_with_embeddings
from process_product import handle_product_sync, handle_webhook

load_dotenv()
# set logging level
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
model = FashionCLIP("fashion-clip")

supabase_key = os.environ.get("SUPABASE_ANON_KEY")
supabase_url = os.environ.get("SUPABASE_URL")
openai_api_key = os.environ.get("OPEN_API_KEY")
api_version = os.environ.get("API_VERSION")
token = os.environ.get("TOKEN")

# Setup OpenAI Client
openai = OpenAI(api_key=openai_api_key)
# Set up Supabase client
supabase_client = create_client(supabase_url, supabase_key)


class TagResponse(BaseModel):
    """tag response model"""

    occasionTags: conlist(str, min_length=1)  # type: ignore
    seasonalTags: conlist(str, min_length=1)  # type: ignore
    styleTags: conlist(str, min_length=1)  # type: ignore
    descriptionAnalysis: conlist(str, min_length=1)  # type: ignore
    colourAndTone: conlist(str, min_length=1)  # type: ignore
    productCategory: str


def fetch_products(url, headers, cursor=None):
    """Adjust the GraphQL query to use the cursor if provided"""
    after_clause = f', after: "{cursor}"' if cursor else ""
    query = f"""
    {{
      products(first: 250{after_clause}, query: "status:active AND published_status:published AND inventory_total:>0") {{
        edges {{
          cursor
          node {{
            id
            title
            description
            tags
            totalInventory
            onlineStoreUrl
            priceRange {{
              maxVariantPrice {{
                amount
              }}
            }}
            featuredImage {{
              url
            }}
            productType
            tags
            vendor
            variants(first: 10) {{
              edges {{
                node {{
                  id
                  price 
                  title
                  inventoryQuantity
                  image {{
                    url
                    
                  }}
                  selectedOptions {{
                    name
                    value
                  }}
                }}
              }}
            }}
          }}
        }}
        pageInfo {{
          hasNextPage
        }}
      }}
    }}
    """
    try:
        response = requests.post(url, headers=headers, json={"query": query})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        raise


def paginate_through_all_products(url, headers):
    """Fetch all products and return them in a list"""
    all_products = []
    cursor = None  # Start with no cursor
    has_next_page = True

    while has_next_page:
        result = fetch_products(cursor=cursor, url=url, headers=headers)

        products_data = result["data"]["products"]
        edges = products_data["edges"]
        for edge in edges:
            all_products.append(edge["node"])
            cursor = edge["cursor"]
        has_next_page = products_data["pageInfo"]["hasNextPage"]
        print(f"running count of products: {len(all_products)}")
    return all_products


def process_variant_data(product_data):
    """Process variant data"""

    processed_variants = []
    # process variants
    if "variants" not in product_data.keys():
        print("Invalid payload - variants not present")
        return 0
    if "edges" not in product_data["variants"].keys():
        print("Invalid variant data - edges not present")

    for node in product_data["variants"]["edges"]:
        processed_data = {}

        if "node" not in node.keys():
            print("Invalid node - key node not present")
            continue
        try:
            processed_data["id"] = node["node"]["id"]
            # check if variant quantity is 0
            if "inventoryQuantity" in node["node"].keys():
                inventory_quantity = node["node"]["inventoryQuantity"]
                if inventory_quantity < 1:
                    product_id = product_data["id"]
                    logging.error(
                        "Skipping variant for node id %s because it is out of stock",
                        product_id,
                    )

            processed_data["price"] = node["node"]["price"]

            if node["node"]["image"] is not None:
                processed_data["image"] = node["node"]["image"]["url"]
            else:
                processed_data["image"] = ""
            variant_details = ""
            for options in node["node"]["selectedOptions"]:
                variant_details += options["name"] + "- " + options["value"] + ","
            processed_data["variant_details"] = variant_details

        except Exception as e:
            logging.error("Error in processing variant data: %s", e)
            raise
        processed_variants.append(processed_data)

    return processed_variants


def generate_tags(product_content):
    """Generate tags for a product"""
    TAGS_PROMPT = f"""
    Your job is to generate a json object of tags and information regarding a product to be used later on for matching outfits of different products together.
    The object should contain the following fields: occasionTags (string array), seasonalTags: (string array), styleTags (string array), descriptionAnalysis (string array), colourAndTone (string array), productCategory (string).
    Here is a breakdown of the following fields:
    - occasionTags: A string array of tags that denote the occasion (e.g., “office,” “casual,” “date night”).
    - seasonalTags: A string array of tags related to the season (e.g., “summer,” “winter”).
    - styleTags: A string array of tags that indicate the style (e.g., “boho,” “classic,” “modern”).
    - descriptionAnalysis: A string array of keywords from item descriptions to understand additional attributes like material, fit, and special features (e.g., “stretchy,” “lightweight”). Do not include information regarding clothing material percentages or cleaning instructions here.
    - colourAndTone: A string array of tags regarding the products color and tone. (e.g., "Neutral", "Neon", "Purple", Beige", "Red", etc...). Make sure you add at least one tag regarding tone in addition to the colour tags.
    - productCategory: A string of the product category (e.g., "Dress", "Shorts", "Pants", "Bottom", "Accessory", "Top").
    Here is the product information: {product_content}
    
    """
    response = openai.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": TAGS_PROMPT},
        ],
    )
    product_id = product_content["id"]
    try:
        print(response.choices[0].message.content)
        tags = json.loads(response.choices[0].message.content)
        print(tags)
        validated_tags = TagResponse(**tags)
        # print validated tags

        print(f"Validated tags: {validated_tags} for product {product_id}")

        return tags

    except Exception as e:
        print(f"Error generating or validating tags {product_id}: {e}")

    return {}


# Decorator to check if the token is provided and valid
def require_token(func):
    """Decorator to check if the token is provided and valid"""

    def wrapper(*args, **kwargs):
        received_token = request.headers.get("Authorization")

        if received_token != f"Bearer {token}":
            return jsonify({"message": "Unauthorized"}), 401

        return func(*args, **kwargs)

    return wrapper


@app.route("/fetch_products_api", endpoint="fetch_products_api", methods=["GET"])
@require_token
def fetch_products_api():
    shop_url = request.args.get("shop_url")
    access_token = request.args.get("access_token")

    if not shop_url or not access_token:
        return jsonify({"error": "Missing shop_url or access_token"}), 400

    url = f"https://{shop_url}/admin/api/{api_version}/graphql.json"
    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Access-Token": access_token,
    }
    try:
        print(f"using url {url}")
        products = paginate_through_all_products(url=url, headers=headers)
        return asyncio.run(handle_product_sync(products, shop_url))
    except Exception as e:
        logging.error("Error processing products: %s", e)
        return (
            jsonify({"error": "Failed to fetch products due to an internal error"}),
            500,
        )


@app.route("/update-products", endpoint="update-products", methods=["POST"])
@require_token
def update_products():
    """Update products in Shopify based on webhook events"""
    data = request.get_json()
    shop = data.get("shop_url")
    access_token = data.get("access_token")
    product = data.get("product")

    if not shop or not access_token or not product:
        return jsonify({"error": "Missing shop_url or access_token"}), 400

    return asyncio.run(handle_webhook(product, shop))


@app.route("/fetch-suggestions", endpoint="fetch-suggestions", methods=["POST"])
@require_token
def fetch_suggestions():
    """Fetch suggestions based on embeddings"""
    data = request.get_json()
    shop_url = data.get("shop_url")
    inputs = data.get(
        "inputs"
    )  # format: [{"item_type": "Dress", "input": "Fall Breezy Dress"}]

    input_texts = [item["input"] for item in inputs]
    item_types = [item["item_type"] for item in inputs]
    encodings = model.encode_text(input_texts, len(input_texts))

    inputs_two = []
    for i in range(len(item_types)):
        inputs_two.append({"embedding": encodings[i], "item_type": item_types[i]})

    recommendations = asyncio.run(get_reccs(shop_url, inputs_two))

    return jsonify(recommendations), 200


async def get_reccs(shop_url, inputs):
    """Fetch recommendations based on embeddings"""
    tasks = [
        recommend_outfits_with_embeddings(
            input["embedding"], shop_url, input["item_type"]
        )
        for input in inputs
    ]
    # Gather results asynchronously
    results = await asyncio.gather(*tasks)
    return results


@app.route("/embed-text", endpoint="embed-text", methods=["POST"])
def embed_text_req():
    """Embed text and return the embedding"""
    body = request.get_json()
    description = body.get("description")
    return jsonify(embed_text(description)), 200


@app.route("/embed-image", endpoint="embed-imagine", methods=["POST"])
def embed_image_req():
    """Embed image and return the embedding"""
    body = request.get_json()
    image = body.get("imageUrl")
    return jsonify(embed_image(image)), 200


@app.route("/")
def hello():
    """test endpoint"""
    return "Hello, Docker!"


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("ENV", "development") != "production",
        port=os.environ.get("PORT", 5000),
    )
