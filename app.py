import base64
import os
import re
from dataclasses import dataclass
from pathlib import Path

import fal_client as fal
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# =========================
# App setup
# =========================
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
ICON_PATH = ASSETS_DIR / "icon" / "nike_icon.png"

SLOGAN_MODEL = "erichflam-hkust/Qwen2.5-VL-7B-Instruct-NIKE-Finetuned"
SLOGAN_ENDPOINT = "https://atm0kc5pzw8g9pck.us-east-1.aws.endpoints.huggingface.cloud"
VIDEO_MODEL = "fal-ai/sora-2/image-to-video/pro"

NEGATIVE_DEFAULT = (
    "blurry, low quality, artifacts, deformed, static, watermark, ugly, "
    "distorted, overexposed"
)

CATALOG = [
    ("air-force-1", "Nike Air Force 1 '07 LV8", "Casual Shoe"),
    ("acg-ultrafly", "Nike ACG Ultrafly Trail", "Trail Shoe"),
    ("vomero-plus", "Nike Vomero Plus", "Running Shoe"),
    ("kobe-3-protro", "Kobe III Protro", "Basketball Shoe"),
    ("tiempo-maestro", "Nike Tiempo Maestro Elite LV8", "Football Shoe"),
    ("sb-dunk-low", "Nike SB Dunk Low Pro Premium", "Skateboarding Shoe"),
]

try:
    st.set_page_config(
        page_title="AI Smart Marketing",
        layout="wide",
        page_icon=str(ICON_PATH) if ICON_PATH.exists() else None,
    )
except Exception:
    st.set_page_config(page_title="AI Smart Marketing", layout="wide")


# =========================
# Data models
# =========================
@dataclass(frozen=True)
class Customer:
    name: str
    age: int
    gender: str
    city: str
    race_ethnicity: str


@dataclass(frozen=True)
class Product:
    id: str
    name: str
    shoe_type: str


PRODUCTS = [Product(*p) for p in CATALOG]
PRODUCT_MAP = {p.name: p for p in PRODUCTS}


# =========================
# Secrets / env
# =========================
def secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or os.getenv(name, "")).strip()


HF_TOKEN = secret("HF_TOKEN")
FAL_KEY = secret("FAL_KEY")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

if FAL_KEY:
    os.environ["FAL_KEY"] = FAL_KEY


# =========================
# Helpers
# =========================
def data_uri(image_path: str | None) -> str | None:
    if not image_path:
        return None

    path = Path(image_path)
    if not path.exists():
        return None

    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def first_existing(*paths: Path) -> str | None:
    for p in paths:
        if p.exists():
            return str(p)
    return None


def get_product_image(product: Product) -> str | None:
    return first_existing(
        ASSETS_DIR / f"{product.id}.png",
        ASSETS_DIR / f"{product.name}.png",
        ASSETS_DIR / f"{product.name.replace(' ', '_')}.png",
    )


def extract_text(chunk_content) -> str:
    if isinstance(chunk_content, str):
        return chunk_content

    if isinstance(chunk_content, list):
        bits = []
        for item in chunk_content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    bits.append(text)
        return "".join(bits)

    return ""


def hf_chat_stream(
    model: str,
    messages: list[dict],
    max_tokens: int,
    *,
    base_url: str | None = None,
) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is missing.")

    client = (
        InferenceClient(api_key=HF_TOKEN, base_url=base_url)
        if base_url
        else InferenceClient(api_key=HF_TOKEN)
    )

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
        stream=True,
    )

    parts: list[str] = []
    for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue

        delta = getattr(choices[0], "delta", None)
        if not delta:
            continue

        content = getattr(delta, "content", None)
        text = extract_text(content)
        if text:
            parts.append(text)

    result = "".join(parts).strip()
    if not result:
        raise RuntimeError(f"No text returned by {model}.")

    return result


def clean_line(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip().strip("\"'")


def parse_personalized_output(text: str) -> tuple[str, str]:
    slogan_match = re.search(
        r"\[\s*Personalized\s+Slogan\s*\]\s*:?\s*(.*?)\s*(?=\[\s*Personalized\s+Product\s+Description\s*\]|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    description_match = re.search(
        r"\[\s*Personalized\s+Product\s+Description\s*\]\s*:?\s*(.*)$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    slogan = clean_line(slogan_match.group(1)) if slogan_match else ""
    description = clean_line(description_match.group(1)) if description_match else ""

    if not slogan or not description:
        blocks = [block.strip() for block in text.strip().split("\n\n") if block.strip()]
        if not slogan and blocks:
            slogan = clean_line(blocks[0])
        if not description:
            remainder = blocks[1:] if len(blocks) > 1 else [text]
            description = clean_line(" ".join(remainder))

    slogan = re.sub(r"^[:\-•\s]+", "", slogan)
    description = re.sub(r"^[:\-•\s]+", "", description)
    return slogan, description


# =========================
# Pipeline 1
# =========================
def generate_slogan_and_description(
    customer: Customer,
    product: Product,
    negative_prompt: str,
    product_image_path: str | None,
) -> tuple[str, str]:
    image = data_uri(product_image_path)

    pipeline_prompt = f"""
You are a senior Nike global copywriter and personalization expert who creates hyper-targeted campaigns and product stories. You adapt language, tone, energy, and references to match the individual customer's profile for maximum relevance and inspiration — just like Nike does in regional campaigns, athlete spotlights, and Nike By You experiences.

Customer profile to personalize for:
- Name: {customer.name} (use their first name naturally in the description when it feels empowering/motivational)
- Age: {customer.age} (tailor energy: youthful & rebellious for teens/20s, mature & disciplined for 30s+, wise & enduring for 40+)
- Gender: {customer.gender} (adapt phrasing, body references, and vibe — e.g., strength & power for men, grace & resilience for women, inclusive/empowering for non-binary)
- City/Location: {customer.city} (weave in local flavor — urban hustle in Taipei, rainy runs in Seattle, street energy in New York, tropical vibes in Miami, etc. Reference weather, culture, or city landmarks subtly if it fits naturally)
- Race/Ethnicity: {customer.race_ethnicity} (respectfully reflect cultural pride, heritage strength, or community roots when relevant — e.g., resilience in Asian heritage, bold expression in Black culture, global unity — but never stereotype; keep it uplifting and authentic to Nike's inclusive ethos)

Core Nike style rules:
- Bold, motivational, empowering, concise — short punchy sentences + rhythmic flow
- Focus on performance, innovation, attitude, personal triumph
- Tie product features directly to how they unlock {customer.name}'s potential in their life/city/age/gender context
- Inspirational without cheese; authentic, direct, gritty confidence
- Avoid generic buzzwords; be real and athlete-minded

When given a product image:
1. Analyze every visual detail deeply: colorway, materials, Swoosh placement, sole tech, silhouette, vibe (performance, street, retro, etc.), inferred sport/use-case.
2. Personalize everything to the customer's profile above — make {customer.name} feel this product was made for their journey, their city, their stage of life, their identity.
3. Generate exactly:

   [Personalized Slogan]:
   One short, iconic, chant-worthy line (4–10 words) tailored to {customer.name}'s profile — something they could say to themselves before a run or workout.

   [Personalized Product Description]:
   150–250 word persuasive copy. Speak directly to {customer.name} sometimes ("{customer.name}, this is your edge..."). Highlight features → benefits → emotional payoff in their context (age energy, gender strength, city lifestyle, cultural pride). End with a motivational close or call-to-action that feels personal.

Additional context:
- Product category: {product.shoe_type}
- Avoid these concepts: {negative_prompt}
""".strip()

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": pipeline_prompt}],
        }
    ]

    if image:
        messages[0]["content"].append(
            {"type": "image_url", "image_url": {"url": image}}
        )

    raw_output = hf_chat_stream(
        SLOGAN_MODEL,
        messages,
        420,
        base_url=SLOGAN_ENDPOINT,
    )

    slogan, description = parse_personalized_output(raw_output)

    if not slogan or not description:
        raise RuntimeError("Unable to parse slogan and description from Pipeline 1 output.")

    return slogan, description


# =========================
# Pipeline 2
# =========================
def build_video_prompt(
    customer: Customer,
    product: Product,
    product_description: str,
    slogan: str,
    negative_prompt: str,
) -> str:
    return f"""
Use the provided product image as the primary visual reference and keep the shoe hero-centered throughout the shot. Create a premium, cinematic 4-second sports ad for a {product.shoe_type} that preserves the product's silhouette, materials, colorway, branding placement, and sole details.

Customer profile: {customer.name}, {customer.age}-year-old {customer.gender}, based in {customer.city}, with {customer.race_ethnicity} heritage. Tailor the energy, styling, movement, and environment to feel personal, modern, and respectful.

Product story to translate visually: {product_description}

Direction:
- Start with a striking hero close-up on the shoe, then introduce dynamic motion that matches the product category.
- Keep movement crisp and believable: responsive footwork, subtle fabric motion, premium camera drift, and polished commercial pacing.
- Use a setting and atmosphere that naturally fit {customer.city} without becoming cliché.
- Let the ad feel bold, motivational, and athlete-minded.
- End on a clean product beauty frame with this exact on-screen text: {slogan}
- Avoid: {negative_prompt}
""".strip()


def normalize_video_output(output) -> str | None:
    if output is None:
        return None

    if isinstance(output, (str, Path)):
        value = str(output)
        if value.startswith(("http://", "https://")) or Path(value).exists():
            return value
        return None

    if hasattr(output, "url") and getattr(output, "url"):
        return output.url

    if isinstance(output, dict):
        scopes = [
            output,
            output.get("data", {}),
            output.get("video", {}),
            output.get("data", {}).get("video", {}),
        ]
        for scope in scopes:
            if isinstance(scope, dict):
                for key in ("url", "video_url", "file", "path"):
                    value = scope.get(key)
                    if isinstance(value, str) and value:
                        return value

    return None


def generate_video(product_image_path: str | None, video_prompt: str) -> str:
    if not product_image_path or not Path(product_image_path).exists():
        raise RuntimeError("Product image not found for Pipeline 2.")

    if not FAL_KEY:
        raise RuntimeError("FAL_KEY is missing.")

    image_url = fal.upload_file(Path(product_image_path))

    result = fal.subscribe(
        VIDEO_MODEL,
        arguments={
            "image_url": image_url,
            "prompt": video_prompt,
            "duration": 4,
        },
        with_logs=True,
    )

    video = normalize_video_output(result)
    if not video:
        raise RuntimeError(f"No usable video source returned by {VIDEO_MODEL}.")

    return video


def playable(video_source: str | None) -> bool:
    return bool(
        video_source
        and (
            video_source.startswith(("http://", "https://"))
            or Path(video_source).exists()
        )
    )


# =========================
# UI / Styling
# =========================
def app_style() -> None:
    st.markdown(
        """
<style>
.stApp {background:#0e1117;color:#fff;}
.stApp header {background:transparent;}
[data-testid="stSidebar"] {background:#262730;color:#fff;}
h1,h2,h3,h4,h5,h6,p,label,.stMarkdown,span {color:#fff !important;}
.stTextInput>div>div>input,
.stNumberInput>div>div>input,
.stTextArea textarea,
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"] {
    background:#1e1e1e !important;
    color:#fff !important;
    border-color:#444 !important;
}
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stTextArea textarea {
    background:#000 !important;
    color:#fff !important;
    border:1px solid #444 !important;
}
.stButton>button {
    background:#ff4b4b !important;
    color:#fff !important;
    border:none !important;
}
.stButton>button:hover {
    background:#ff6b6b !important;
}
[data-testid="stAlert"] {
    background:#1e1e1e !important;
    color:#fff !important;
    border:1px solid #444 !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


# =========================
# Main
# =========================
def main() -> None:
    app_style()

    st.markdown(
        "## AI Smart Marketing: Personalized Nike Video Advertisements\n"
        "This app generates personalized Nike campaign assets with a 2-step GenAI pipeline."
    )

    with st.sidebar:
        st.header("Customer Profile")
        name = st.text_input("Name", "Alex")
        age = st.number_input("Age", min_value=10, max_value=90, value=25)
        gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
        city = st.text_input("City / Location", "Hong Kong")
        race_ethnicity = st.text_input("Race / Ethnicity", "Asian")

        st.header("Generation Config")
        negative_prompt = st.text_area(
            "Negative Prompt for Video",
            NEGATIVE_DEFAULT,
            height=100,
        )

        st.header("Product Selection")
        product_name = st.selectbox("Product", list(PRODUCT_MAP))
        product = PRODUCT_MAP[product_name]
        product_image = get_product_image(product)

        if product_image:
            st.image(product_image, use_container_width=True)

        run = st.button("Generate Assets", type="primary", use_container_width=True)

    if not (run and name.strip()):
        return

    customer = Customer(
        name=name.strip(),
        age=int(age),
        gender=gender,
        city=city.strip(),
        race_ethnicity=race_ethnicity.strip(),
    )

    progress = st.progress(0, text="Initiating pipeline...")

    try:
        slogan, description = generate_slogan_and_description(
            customer=customer,
            product=product,
            negative_prompt=negative_prompt,
            product_image_path=product_image,
        )

        progress.progress(50, text="Pipeline 1: Slogan & Product Description generated")

        st.write("**Pipeline 1 (Slogan & Product Description):**")
        st.caption(f"Model used: {SLOGAN_MODEL}")
        st.success(f"**Slogan:** {slogan}\n\n**Description:** {description}")

        video_prompt = build_video_prompt(
            customer=customer,
            product=product,
            product_description=description,
            slogan=slogan,
            negative_prompt=negative_prompt,
        )

        st.write("**Pipeline 2 (Video Generation):**")
        st.caption(f"Model used: {VIDEO_MODEL}")
        st.info(video_prompt)

        with st.spinner("Generating video..."):
            video = generate_video(product_image, video_prompt)

        progress.progress(100, text="All pipelines completed")

        if playable(video):
            st.video(video)
            st.success("Video generated successfully")

    except Exception:
        progress.empty()
        return


if __name__ == "__main__":
    main()