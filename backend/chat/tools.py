"""Tools the model may call. Price and stock live here, never in the embeddings (K-5)."""
from difflib import SequenceMatcher

from django.utils.text import slugify

from catalog.models import Product

CANDIDATE_LIMIT = 3

_SLUG_PARAM = {
    "type": "object",
    "properties": {
        "product_slug": {
            "type": "string",
            "description": "Ürünün slug değeri; <parca> etiketindeki slug özniteliğinden alınır.",
        },
    },
    "required": ["product_slug"],
}

TOOL_DEFINITIONS = [
    {
        "name": "get_price",
        "description": "Bir ürünün güncel satış fiyatını (TL) veritabanından getirir.",
        "parameters": _SLUG_PARAM,
    },
    {
        "name": "check_stock",
        "description": "Bir ürünün güncel stok adedini ve stokta olup olmadığını veritabanından getirir.",
        "parameters": _SLUG_PARAM,
    },
]


def _candidates(query: str) -> list[dict]:
    """Products closest to an unknown slug, best first."""
    q = slugify(query)
    q_tokens = set(q.split("-")) - {""}
    scored = []
    for p in Product.objects.only("slug", "name"):
        name_slug = slugify(p.name)
        tokens = set(p.slug.split("-")) | set(name_slug.split("-"))
        overlap = len(q_tokens & tokens) / len(q_tokens) if q_tokens else 0.0
        score = max(
            SequenceMatcher(None, q, p.slug).ratio(),
            SequenceMatcher(None, q, name_slug).ratio(),
            0.9 * overlap,
        )
        scored.append((score, p))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [{"slug": p.slug, "name": p.name} for _, p in scored[:CANDIDATE_LIMIT]]


def _lookup(product_slug) -> tuple[Product | None, dict | None]:
    """The product, or an error result that lists the closest candidates."""
    slug = product_slug.strip() if isinstance(product_slug, str) else ""
    product = Product.objects.filter(slug=slug).first() if slug else None
    if product:
        return product, None
    return None, {
        "error": f"'{slug}' slug değerine sahip ürün bulunamadı.",
        "candidates": _candidates(slug),
        "hint": "Adaylardan uygun olanın slug değeriyle aracı yeniden çağır; "
                "uygun aday yoksa müşteriye ürünü bulamadığını söyle.",
    }


def get_price(product_slug) -> dict:
    product, error = _lookup(product_slug)
    if error:
        return error
    return {"slug": product.slug, "name": product.name, "price": f"{product.price:.2f}", "currency": "TRY"}


def check_stock(product_slug) -> dict:
    product, error = _lookup(product_slug)
    if error:
        return error
    return {"slug": product.slug, "name": product.name, "in_stock": product.stock > 0, "stock": product.stock}


_TOOLS = {"get_price": get_price, "check_stock": check_stock}


def run_tool(name: str, args: dict) -> dict:
    """Execute a tool by name; never raises, so a bad model call becomes a tool result."""
    func = _TOOLS.get(name)
    if func is None:
        return {"error": f"Bilinmeyen araç: {name}", "available": sorted(_TOOLS)}
    try:
        return func(**args)
    except TypeError:
        return {"error": f"{name} aracı için geçersiz argümanlar.", "expected": ["product_slug"]}
