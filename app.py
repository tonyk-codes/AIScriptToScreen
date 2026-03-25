import base64, os, re
from dataclasses import dataclass
from pathlib import Path
import fal_client as fal, streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# App setup
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
ICON_PATH = ASSETS_DIR / "icon" / "nike_icon.png"

SLOGAN_MODEL = "erichflam-hkust/Qwen2.5-VL-7B-Instruct-NIKE-Finetuned"
SLOGAN_ENDPOINT = "https://atm0kc5pzw8g9pck.us-east-1.aws.endpoints.huggingface.cloud"
SCRIPT_MODEL = "zai-org/GLM-4.7-Flash:novita"
VIDEO_MODEL = "fal-ai/ltx-2.3/image-to-video"

NATIONALITIES = ["Chinese", "American", "Indian", "Indonesian", "Pakistani", "Nigerian", "Brazilian", "Bangladeshi", "Russian", "Mexican"]
NEGATIVE_DEFAULT = "blurry, low quality, artifacts, deformed, static, watermark, ugly, distorted, overexposed"
CATALOG = [
    ("air-force-1", "Nike Air Force 1 '07 LV8", "Casual Shoe"), ("acg-ultrafly", "Nike ACG Ultrafly Trail", "Trail Shoe"),
    ("vomero-plus", "Nike Vomero Plus", "Running Shoe"), ("kobe-3-protro", "Kobe III Protro", "Basketball Shoe"),
    ("tiempo-maestro", "Nike Tiempo Maestro Elite LV8", "Football Shoe"), ("sb-dunk-low", "Nike SB Dunk Low Pro Premium", "Skateboarding Shoe"),
]

try:
    st.set_page_config(page_title="AI Smart Marketing", layout="wide", page_icon=str(ICON_PATH) if ICON_PATH.exists() else None)
except Exception:
    st.set_page_config(page_title="AI Smart Marketing", layout="wide")

@dataclass(frozen=True)
class Customer:
    name: str; age: int; gender: str; nationality: str

@dataclass(frozen=True)
class Product:
    id: str; name: str; shoe_type: str

PRODUCTS = [Product(*p) for p in CATALOG]
PRODUCT_MAP = {p.name: p for p in PRODUCTS}

def secret(name: str) -> str:
    """Read a secret with one simple path only."""
    try: value = st.secrets.get(name, "")
    except Exception: value = ""
    return str(value or os.getenv(name, "")).strip()

HF_TOKEN = secret("HF_TOKEN")
FAL_KEY = secret("FAL_KEY")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
if FAL_KEY:
    os.environ["FAL_KEY"] = FAL_KEY

def data_uri(image_path: str | None) -> str | None:
    if not image_path or not (path := Path(image_path)).exists(): return None
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('utf-8')}"

def first_existing(*paths: Path) -> str | None:
    return next((str(p) for p in paths if p.exists()), None)

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

def hf_chat_stream(model: str, messages: list[dict], max_tokens: int, *, base_url: str | None = None) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is missing.")
    client = InferenceClient(api_key=HF_TOKEN, base_url=base_url) if base_url else InferenceClient(api_key=HF_TOKEN)
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

def hf_chat_once(model: str, messages: list[dict], max_tokens: int, *, base_url: str | None = None) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is missing.")
    client = InferenceClient(api_key=HF_TOKEN, base_url=base_url) if base_url else InferenceClient(api_key=HF_TOKEN)
    result = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=0.7)
    choices = getattr(result, "choices", None) or []
    if not choices: raise RuntimeError(f"No text returned by {model}.")
    text = extract_text(getattr(choices[0].message, "content", None))
    if not text: raise RuntimeError(f"No text returned by {model}.")
    return text

def clean_slogan(text: str, customer_name: str) -> str:
    """Light post-processing to enforce the requested slogan format."""
    text = re.sub(r"[.!?]+$", "", re.sub(r"\bNike\b", "", re.sub(r"\s+", " ", text.replace("\n", " ")).strip().strip('"\''), flags=re.I).strip(" ,")).strip()
    head = re.sub(rf",?\s*{re.escape(customer_name)}$", "", text, flags=re.I).strip(" ,") if text.lower().endswith(customer_name.lower()) else text
    head = " ".join(head.split()[:10])
    return f"{head}, {customer_name}".strip(" ,") + ("" if head.endswith(customer_name) else "")

def generate_slogan_and_description(
    customer: Customer,
    product: Product,
    negative_prompt: str,
    product_image_path: str | None,
) -> tuple[str, str]:
    image = data_uri(product_image_path)

    # Slogan Generation
    slogan_prompt = f"""
You are an expert copywriter. Write a short, natural-sounding ad sentence for a {customer.age}-year-old {customer.gender} from {customer.nationality} buying {product.shoe_type}.

CRITICAL INSTRUCTIONS:
- Write in normal English language with regular spaces between words.
- Maximum 9 words total.
- Do NOT include labels like "Slogan:" , "—", "." , or quotes.
- Do NOT use the words "Nike", "Just do it", or specific model names.
- The text must end exactly with: , {customer.name} (with no period at the end).

EXAMPLE 1:
Walk with power and style everywhere you go, {customer.name}

EXAMPLE 2:
Your daily run just got a lot smoother, {customer.name}

Avoid these concepts: {negative_prompt}

Output the exact slogan now:
""".strip()

    slogan_messages = [{"role": "user", "content": [{"type": "text", "text": slogan_prompt}]}]
    if image:
        slogan_messages[0]["content"].append({"type": "image_url", "image_url": {"url": image}})
    
    # Generate the slogan text from the stream
    raw_slogan = hf_chat_stream(SLOGAN_MODEL, slogan_messages, 80, base_url=SLOGAN_ENDPOINT)
    
    # Ensure any stray whitespace or newlines are stripped only at the very end
    # Assuming clean_slogan() is your custom function that formats the name suffix
    slogan = clean_slogan(raw_slogan.strip(), customer.name)

    # Description Generation
    description_prompt = f"""
You are an expert copywriter. Write exactly TWO vivid marketing sentences for a {product.shoe_type}.

CRITICAL RULES:
1. Write in normal English with regular spaces between words.
2. Output exactly two sentences total in a single paragraph. No numbered lists (e.g., "1.", "2.").
3. Seamlessly integrate the target customer: {customer.age}-year-old {customer.nationality} {customer.gender}.
4. Describe the performance, design, and fit based on the {product.shoe_type} and image.
5. DO NOT use specific brand names (like Nike) or product model names (like Pegasus).
6. DO NOT use the customer's name.
7. Output ONLY the two sentences with no introductory text.

EXAMPLE FORMAT:
Designed for the active 28-year-old Japanese woman, these lightweight running shoes offer unmatched breathability and a responsive, cloud-like midsole. Whether you are sprinting through city streets or enjoying a casual jog, the sleek cinematic design ensures a secure, locked-in fit that matches your relentless pace.

Avoid these concepts: {negative_prompt}

Output the 2 sentences now:
""".strip()

    description_messages = [{"role": "user", "content": [{"type": "text", "text": description_prompt}]}]
    if image:
        description_messages[0]["content"].append({"type": "image_url", "image_url": {"url": image}})
    
    # Generate the description text from the stream
    raw_description = hf_chat_stream(SLOGAN_MODEL, description_messages, 180, base_url=SLOGAN_ENDPOINT)
    
    # Strip whitespace only on the final assembled string
    description = raw_description.strip()
    
    return slogan, description

def generate_cinematic_script(
    customer: Customer,
    product: Product,
    product_description: str,
    slogan: str,
    negative_prompt: str,
) -> str:
    system_prompt = f"""
You are an elite image-to-video prompt engineer for premium sports ads.
Write one complete, production-ready prompt for a video model. Finish every section cleanly.

Use this exact structure and no other text:
[Subject / Hero Shot]:
[Scene & Environment]:
[Motion & Dynamics]:
[Camera & Cinematography]:
[Lighting & Mood]:
[Personalization Layer]:
[End Frame & On-Screen Text]:
[Style & Quality Boosters]:

Requirements:
- The product image will be provided separately to the video model as the visual reference.
- Keep the {product.shoe_type} in clear focus throughout.
- Build a coherent ad, optimized for script-to-video generation.
- Make camera direction precise and executable.
- Use vivid but concise cinematic language.
- Tailor energy, styling, and motion to a {customer.age}-year-old {customer.gender} customer from {customer.nationality}.
- The final section must show this exact on-screen slogan at the end of the video: {slogan}
- Do not mention the product model name.
- Avoid: {negative_prompt}
""".strip()

    user_prompt = f"""
Target customer: {customer.age}, {customer.gender}, {customer.nationality}
Shoe type: {product.shoe_type}
Product description: {product_description}
Generate the full script now.
""".strip()

    return hf_chat_once(SCRIPT_MODEL, [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], 4000)

def normalize_video_output(output) -> str | None:
    if output is None:
        return None
    if isinstance(output, (str, Path)):
        value = str(output)
        return value if value.startswith(("http://", "https://")) or Path(value).exists() else None
    if hasattr(output, "url") and getattr(output, "url"):
        return output.url
    if isinstance(output, dict):
        for scope in (output, output.get("data", {}), output.get("video", {}), output.get("data", {}).get("video", {})):
            if isinstance(scope, dict):
                for key in ("url", "video_url", "file", "path"):
                    value = scope.get(key)
                    if isinstance(value, str) and value:
                        return value
    return None

def generate_video(product_image_path: str | None, cinematic_script: str, slogan: str) -> str:
    if not product_image_path or not Path(product_image_path).exists():
        raise RuntimeError("Product image not found for Pipeline 3.")
    if not FAL_KEY:
        raise RuntimeError("FAL_KEY is missing.")

    image_url = fal.upload_file(Path(product_image_path))
    prompt = (
        "Use the provided product image as the visual reference.\n"
        "Follow this script closely and keep the shoe prominent.\n"
        f"{cinematic_script}\n"
        f'End the video with the exact on-screen slogan "{slogan}", presented elegantly in a stylish, cinematic composition.'
    )

    result = fal.subscribe(VIDEO_MODEL, arguments={"image_url": image_url, "prompt": prompt, "generate_audio": False, "duration": 6}, with_logs=True)
    video = normalize_video_output(result)
    if not video:
        raise RuntimeError(f"No usable video source returned by {VIDEO_MODEL}.")
    return video

def playable(video_source: str | None) -> bool:
    return bool(video_source and (video_source.startswith(("http://", "https://")) or Path(video_source).exists()))

def app_style() -> None:
    st.markdown("""<style>
        .stApp {background:#0e1117;color:#fff;} .stApp header {background:transparent;}
        [data-testid="stSidebar"] {background:#262730;color:#fff;}
        h1,h2,h3,h4,h5,h6,p,label,.stMarkdown,span {color:#fff !important;}
        .stTextInput>div>div>input,.stNumberInput>div>div>input,.stTextArea textarea,div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {background:#1e1e1e !important;color:#fff !important;border-color:#444 !important;}
        [data-testid="stSidebar"] .stNumberInput input,[data-testid="stSidebar"] .stTextArea textarea {background:#000 !important;color:#fff !important;border:1px solid #444 !important;}
        .stButton>button {background:#ff4b4b !important;color:#fff !important;border:none !important;} .stButton>button:hover {background:#ff6b6b !important;}
        [data-testid="stAlert"] {background:#1e1e1e !important;color:#fff !important;border:1px solid #444 !important;}
        </style>""", unsafe_allow_html=True)

def main() -> None:
    app_style()
    st.markdown(
        "## AI Smart Marketing: Personalized Nike Video Advertisements\n"
        "This app generates personalized Nike campaign assets with a 3-step GenAI pipeline."
    )

    with st.sidebar:
        st.header("Customer Profile")
        name = st.text_input("Name", "Alex")
        age = st.number_input("Age", 10, 90, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        nationality = st.selectbox("Nationality", NATIONALITIES, index=0)

        st.header("Generation Config")
        negative_prompt = st.text_area("Negative Prompt for Video", NEGATIVE_DEFAULT, height=100)
        
        st.header("Product Selection")
        product_name = st.selectbox("Product", list(PRODUCT_MAP))
        product = PRODUCT_MAP[product_name]
        product_image = get_product_image(product)
        if product_image:
            st.image(product_image, use_container_width=True)
        else:
            st.info(f"No image found for {product.name}")

        if not HF_TOKEN:
            st.warning("HF_TOKEN is missing.")
        if not FAL_KEY:
            st.warning("FAL_KEY is missing.")
        run = st.button("Generate Assets", type="primary", use_container_width=True)

    if not (run and name):
        return

    customer = Customer(name, int(age), gender, nationality)
    progress = st.progress(0, "Initiating pipeline...")

    try:
        slogan, description = generate_slogan_and_description(
            customer, product, negative_prompt, product_image
        )
        progress.progress(25, "Pipeline 1: Slogan & Product Description generated")
        st.write("**Pipeline 1 (Slogan & Product Description):**")
        st.caption(f"Model used: {SLOGAN_MODEL}")
        st.success(f"**Slogan:** {slogan}\n\n**Description:** {description}")

        script = generate_cinematic_script(
            customer, product, description, slogan, negative_prompt
        )
        progress.progress(50, "Pipeline 2: Cinematic script generated")
        st.write("**Pipeline 2 (Cinematic Script Generation):**")
        st.caption(f"Model used: {SCRIPT_MODEL}")
        st.info(script)

        st.write("**Pipeline 3 (Video Generation with End-Screen Slogan):**")
        st.caption(f"Model used: {VIDEO_MODEL}")
        with st.spinner("Generating video..."):
            video = generate_video(product_image, script, slogan)
        progress.progress(100, "All pipelines completed")

        if playable(video):
            st.video(video)
            st.success("Video generated successfully")
        else:
            st.error("Returned source is not playable.")
    except Exception as e:
        st.error(str(e))

if __name__ == "__main__":
    main()
