"""utils script."""

import requests

image_embedding_map = {}


def get_image_embedding(image_url):
    """Get image embedding"""
    image_embedding = []
    if image_url not in image_embedding_map:
        response = requests.post(
            "https://populatedb-production.up.railway.app/embed-image",
            json={"imageUrl": image_url},
            headers={"Content-Type": "application/json"},
        )

        image_embedding = response.json()
    else:
        print(f"Skipped image embed for {image_url} (cache)")

    return image_embedding


def get_text_embedding(description):
    """Get text embedding"""
    response = requests.post(
        "https://populatedb-production.up.railway.app/embed-text",
        json={"description": f"{description}"},
        headers={"Content-Type": "application/json"},
    )

    text_embedding = response.json()
    return text_embedding
