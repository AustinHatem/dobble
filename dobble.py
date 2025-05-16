import streamlit as st
import math
import random
import tempfile
import os
import io
import json
import uuid
import datetime
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
import base64
import pyrebase
import itertools

# Try to load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Set up a flag to track if Firebase is available
FIREBASE_AVAILABLE = False

# Try to import pyrebase with better error handling
try:
    import pyrebase

    FIREBASE_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing pyrebase: {str(e)}")
    st.info("Some features requiring Firebase will be disabled.")


def get_env(key: str) -> str:
    """Get environment variable with fallback to Streamlit secrets."""
    # Check if we can access st.secrets
    try:
        if key.startswith("FIREBASE_"):
            # Convert FIREBASE_API_KEY to api_key format for nested secrets
            firebase_key = key[9:].lower()
            return st.secrets["firebase"][firebase_key]
        else:
            return st.secrets[key]
    except (KeyError, AttributeError):
        # Fallback to environment variables
        value = os.getenv(key)
        if not value:
            st.error(f"❌ Missing environment variable: {key}")
        return value or ""


def initialize_firebase():
    """Initialize Firebase with better error handling"""
    if not FIREBASE_AVAILABLE:
        st.error("Firebase functionality is not available due to import errors.")
        return None, None

    try:
        firebaseConfig = {
            "apiKey": get_env("FIREBASE_API_KEY"),
            "authDomain": get_env("FIREBASE_AUTH_DOMAIN"),
            "projectId": get_env("FIREBASE_PROJECT_ID"),
            "storageBucket": get_env("FIREBASE_STORAGE_BUCKET"),
            "messagingSenderId": get_env("FIREBASE_MESSAGING_SENDER_ID"),
            "appId": get_env("FIREBASE_APP_ID"),
            "measurementId": get_env("FIREBASE_MEASUREMENT_ID"),
            "databaseURL": get_env("FIREBASE_DATABASE_URL"),
        }

        firebase = pyrebase.initialize_app(firebaseConfig)
        auth = firebase.auth()
        auth.sign_in_anonymous()
        storage = firebase.storage()
        db = firebase.database()

        return storage, db
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {str(e)}")
        return None, None


class DobbleGenerator:
    def __init__(
        self,
        output_pdf="dobble_cards.pdf",
        symbols_per_card=6,
        custom_images=None,
    ):
        self.output_pdf = output_pdf
        # Fixed values for cards per row/column
        self.cards_per_row = 2
        self.cards_per_col = 2
        self.symbols_per_card = symbols_per_card
        self.custom_images = custom_images or []

        self.order = self.symbols_per_card - 1
        self.total_symbols_needed = self.order**2 + self.order + 1

        # Validate enough images are available
        if len(self.custom_images) < self.total_symbols_needed:
            raise ValueError(
                f"Not enough custom images. Need at least {self.total_symbols_needed} images."
            )
        self.symbols = self.custom_images[: self.total_symbols_needed]

    def generate_dobble_cards(self):
        """
        Generate a set of Dobble cards using projective geometry principles
        to ensure each pair of cards shares exactly one symbol.

        This implementation is based on a prime power construction
        for a projective plane of order n-1.
        """
        n = self.symbols_per_card
        order = n - 1  # The projective plane order

        # Validate n is valid (n should be a prime power + 1)
        # For simplicity we just check a few common values
        valid_n_values = [3, 4, 6, 8, 12]
        if n not in valid_n_values:
            return (
                None,
                f"Warning: {n} symbols per card might not work with this algorithm.",
            )

        # Formula: For n symbols per card, we need n(n-1)+1 total symbols
        total_symbols = n * (n - 1) + 1

        if total_symbols != self.total_symbols_needed:
            return (
                None,
                f"Expected {self.total_symbols_needed} symbols, but formula requires {total_symbols}",
            )

        # Create empty deck of cards
        cards = []

        # First card has symbols 0 to n-1
        first_card = list(range(n))
        cards.append(first_card)

        # For each point i, create a card with symbol 0 and n-1 other symbols
        for i in range(order):
            card = [0]  # Symbol 0 is common to these cards
            for j in range(order):
                symbol = n + i * order + j
                card.append(symbol)
            cards.append(card)

        # For each line j, create order cards
        for i in range(1, order + 1):
            for j in range(order):
                card = [i]  # Each card contains one of symbols 1 to order
                for k in range(order):
                    # This is the mathematical pattern that ensures each pair
                    # of cards has exactly one symbol in common
                    symbol = n + ((j + k * i) % order) + (k * order)
                    card.append(symbol)
                cards.append(card)

        # Validate card generation - each pair should share exactly one symbol
        invalid_pairs = 0
        for i, card1 in enumerate(cards):
            for j, card2 in enumerate(cards[i + 1 :], i + 1):
                common = set(card1).intersection(set(card2))
                if len(common) != 1:
                    invalid_pairs += 1

        validation_msg = ""
        if invalid_pairs == 0:
            validation_msg = f"✅ All card pairs have exactly one common symbol!"
        else:
            validation_msg = f"❌ Found {invalid_pairs} invalid pairs out of {len(cards)*(len(cards)-1)//2} total pairs"

        # Convert numeric indices to actual symbols
        symbol_cards = []
        for card in cards:
            symbol_card = [self.symbols[symbol] for symbol in card]
            symbol_cards.append(symbol_card)

        status = f"Generated {len(cards)} cards with {self.symbols_per_card} symbols each. {validation_msg}"
        return symbol_cards, status

    def create_dobble_pdf(self):
        cards, status = self.generate_dobble_cards()

        if not cards:
            return False, status

        c = canvas.Canvas(self.output_pdf, pagesize=A4)
        width, height = A4

        card_width = width / self.cards_per_row
        card_height = height / self.cards_per_col

        card_index = 0
        page_number = 1

        while card_index < len(cards):
            for row in range(self.cards_per_col):
                for col in range(self.cards_per_row):
                    if card_index >= len(cards):
                        break

                    card = cards[card_index]
                    x = col * card_width
                    y = height - (row + 1) * card_height

                    c.rect(
                        x + 0.5 * cm,
                        y + 0.5 * cm,
                        card_width - 1 * cm,
                        card_height - 1 * cm,
                    )

                    self._draw_card_symbols(
                        c,
                        card,
                        x + 0.5 * cm,
                        y + 0.5 * cm,
                        card_width - 1 * cm,
                        card_height - 1 * cm,
                    )
                    card_index += 1

            if card_index < len(cards):
                c.showPage()
                page_number += 1

        c.save()
        return (
            True,
            f"✅ PDF saved: {self.output_pdf} with {page_number} page(s). {status}",
        )

    def _draw_card_symbols(self, canvas, symbols, x, y, width, height):
        # Maximize symbol size without overlap
        symbols_per_card = len(symbols)

        # Choose a generous size factor
        size_factor = 0.27  # Try 0.3 or higher if fewer symbols per card

        # Positioning in a circle
        positions = []
        sizes = []

        if symbols_per_card == 1:
            positions.append((0.5, 0.5))
            sizes.append(size_factor * 1.2)
        else:
            radius = 0.35  # Controls how far symbols are from center
            for i in range(symbols_per_card):
                angle = 2 * math.pi * i / symbols_per_card
                pos_x = 0.5 + radius * math.cos(angle)
                pos_y = 0.5 + radius * math.sin(angle)
                positions.append((pos_x, pos_y))
                sizes.append(size_factor)

        # Randomize positions to add variety
        rotations = [random.randint(0, 359) for _ in range(len(symbols))]
        combined = list(zip(positions, sizes))
        random.shuffle(combined)
        positions, sizes = zip(*combined)

        # Draw each image
        for i, symbol in enumerate(symbols):
            rel_x, rel_y = positions[i]
            size = min(width, height) * sizes[i]
            center_x = x + rel_x * width
            center_y = y + rel_y * height
            self._draw_image(canvas, symbol, center_x, center_y, size, rotations[i])

    def _draw_image(self, canvas, image_data, x, y, size, rotation=0):
        """Draw a custom image on the canvas"""
        canvas.saveState()
        canvas.translate(x, y)
        canvas.rotate(rotation)

        # Size is the diameter/width of the symbol
        half_size = size / 2

        # ReportLab expects an image path or a PIL Image
        # Since we have the image data from memory, we'll create a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_filename = temp_file.name

        # Save the PIL image to the temp file
        image_data.save(temp_filename, format="PNG")

        # Draw the image centered at the position
        canvas.drawImage(
            temp_filename,
            -half_size,  # x position (centered)
            -half_size,  # y position (centered)
            size,  # width
            size,  # height
            mask="auto",  # handles transparency
        )

        # Clean up the temp file
        os.unlink(temp_filename)

        canvas.restoreState()


# Helper function to check if two PIL images are similar
def similar_images(img1, img2):
    """
    Check if two PIL images are the same image.
    Returns True if images are similar, False otherwise.
    """
    # Convert to RGB if they're not
    if img1.mode != "RGB":
        img1 = img1.convert("RGB")
    if img2.mode != "RGB":
        img2 = img2.convert("RGB")

    # Resize to a small size for quicker comparison
    size = (20, 20)
    img1_small = img1.resize(size)
    img2_small = img2.resize(size)

    # Get the image data
    img1_data = list(img1_small.getdata())
    img2_data = list(img2_small.getdata())

    # Compare image data directly
    return img1_data == img2_data


# Function to create a download button for the PDF
def get_binary_file_downloader_html(pdf_path, filename):
    """Generate a button to download the pdf file"""
    with open(pdf_path, "rb") as file:
        pdf_contents = file.read()

    b64_pdf = base64.b64encode(pdf_contents).decode()
    download_button_html = f"""
        <a href="data:application/pdf;base64,{b64_pdf}" download="{filename}">
            <button style="background-color: #4CAF50; color: white; padding: 12px 20px; 
            border: none; border-radius: 4px; cursor: pointer; font-size: 16px;">
                Download PDF
            </button>
        </a>
    """
    return download_button_html


# Function to preprocess uploaded images
def preprocess_image(uploaded_file):
    """Process the uploaded image to prepare it for the Dobble card"""
    image = Image.open(uploaded_file)

    # Convert to RGBA if not already
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Make the image square by cropping or padding
    width, height = image.size
    size = max(width, height)
    new_img = Image.new("RGBA", (size, size), (255, 255, 255, 0))

    # Paste the original image centered in the new square image
    paste_x = (size - width) // 2
    paste_y = (size - height) // 2
    new_img.paste(image, (paste_x, paste_y))

    # Resize to a standard size for consistency
    new_img = new_img.resize((300, 300), Image.LANCZOS)

    return new_img


# Check if pyrebase4 is installed, and install if necessary
def check_and_install_dependencies():
    try:
        import pyrebase
    except ImportError:
        import subprocess
        import sys

        st.warning("Installing required dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyrebase4"])
        st.success("Dependencies installed successfully!")
        import pyrebase


def delete_game_from_firebase(storage, db, game_id, image_urls):
    """
    Delete game images from Firebase Storage and metadata from the database.

    This version focuses on deleting the database record first, which is more
    reliable. Since the Storage deletion is causing issues with the Pyrebase library,
    we'll make a best effort to delete images but not fail the entire operation if
    they can't be deleted.
    """
    try:
        # First, remove game metadata from database
        # This ensures that at least the game entry is removed
        db.child("games").child(game_id).remove()

        # At this point, we've successfully deleted the database entry
        # Return success even if we can't delete the images
        return (
            True,
            f"Successfully deleted game '{game_id}' from database. Note: Associated images may still exist in storage.",
        )

    except Exception as e:
        # If database deletion fails, return the error
        return False, f"Error deleting game '{game_id}' from database: {str(e)}"


def initialize_firebase():

    firebaseConfig = {
        "apiKey": get_env("FIREBASE_API_KEY"),
        "authDomain": get_env("FIREBASE_AUTH_DOMAIN"),
        "projectId": get_env("FIREBASE_PROJECT_ID"),
        "storageBucket": get_env("FIREBASE_STORAGE_BUCKET"),
        "messagingSenderId": get_env("FIREBASE_MESSAGING_SENDER_ID"),
        "appId": get_env("FIREBASE_APP_ID"),
        "measurementId": get_env("FIREBASE_MEASUREMENT_ID"),
        "databaseURL": get_env("FIREBASE_DATABASE_URL"),
    }

    firebase = pyrebase.initialize_app(firebaseConfig)
    auth = firebase.auth()
    auth.sign_in_anonymous()
    storage = firebase.storage()
    db = firebase.database()

    return storage, db


# Function to save image to Firebase Storage
def save_image_to_firebase(storage, image, game_id, image_id):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_filename = temp_file.name
        image.save(temp_filename, format="PNG")

    storage_path = f"games/{game_id}/images/{image_id}.png"

    # Try uploading
    try:
        storage.child(storage_path).put(temp_filename)
        download_url = storage.child(storage_path).get_url(None)
    except Exception as e:
        os.unlink(temp_filename)
        raise RuntimeError(f"Failed to upload image to Firebase: {e}")

    os.unlink(temp_filename)
    return download_url


# Function to save game metadata to Firebase
def save_game_to_firebase(db, game_id, game_data):
    db.child("games").child(game_id).set(game_data)


# Function to get all saved games from Firebase
def get_saved_games(db):
    games = db.child("games").get()
    if games.each():
        return [game.val() for game in games.each()]
    return []


# Function to download image from URL
def download_image_from_url(url):
    try:
        import requests
        from io import BytesIO

        response = requests.get(url)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            # Return placeholder image if download fails
            return Image.new("RGBA", (300, 300), color=(255, 0, 0, 128))
    except Exception as e:
        st.warning(f"Error downloading image: {str(e)}")
        # Return placeholder image if download fails
        return Image.new("RGBA", (300, 300), color=(255, 0, 0, 128))


# Function to generate a PDF for a saved game
def generate_pdf_for_saved_game(storage, game_data):
    # Get image download URLs from game data
    image_urls = game_data.get("image_urls", [])

    # Download images from URLs
    with st.spinner("Downloading images..."):
        temp_images = []
        for url in image_urls:
            img = download_image_from_url(url)
            temp_images.append(img)

    # Use the DobbleGenerator to create a PDF
    temp_dir = tempfile.gettempdir()
    output_filename = f"{game_data['title']}.pdf"
    output_path = os.path.join(temp_dir, output_filename)

    try:
        # Generate the cards using the same number of symbols per card as stored
        generator = DobbleGenerator(
            output_pdf=output_path,
            symbols_per_card=game_data["symbols_per_card"],
            custom_images=temp_images[: game_data["total_symbols_needed"]],
        )

        success, _ = generator.create_dobble_pdf()
        if success:
            return output_path, output_filename
        else:
            return None, None
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None, None


# Add a new function to generate card images for the game
def generate_game_cards(game_data, num_cards=None):
    """
    Generate card images for playing the game.
    Returns a list of PIL images representing the cards.
    """
    # Get image download URLs from game data
    image_urls = game_data.get("image_urls", [])
    symbols_per_card = game_data.get("symbols_per_card", 6)
    order = symbols_per_card - 1
    total_symbols_needed = order**2 + order + 1

    # Download images from URLs
    temp_images = []
    for url in image_urls:
        img = download_image_from_url(url)
        temp_images.append(img)

    # Ensure we have enough images
    if len(temp_images) < total_symbols_needed:
        return (
            None,
            f"Not enough images. Need {total_symbols_needed} but got {len(temp_images)}.",
        )

    # Create a generator instance
    generator = DobbleGenerator(
        symbols_per_card=symbols_per_card,
        custom_images=temp_images[:total_symbols_needed],
    )

    # Generate the cards
    cards, status = generator.generate_dobble_cards()

    if not cards:
        return None, status

    # If num_cards is specified, randomly select that many cards
    if num_cards and num_cards < len(cards):
        cards = random.sample(cards, num_cards)

    return cards, status


# Function to draw a card as an image
def draw_card_as_image(symbols, size=600):
    from PIL import ImageDraw

    card_image = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(card_image)

    symbols_per_card = len(symbols)

    # 🔧 LARGER SYMBOLS!
    size_factor = (
        0.33 if symbols_per_card <= 3 else 0.28 if symbols_per_card <= 5 else 0.24
    )

    # 🎯 Adjust radius to spread images more
    radius = 0.35 if symbols_per_card <= 4 else 0.33

    # Calculate positions
    positions = []
    sizes = []
    for i in range(symbols_per_card):
        angle = 2 * math.pi * i / symbols_per_card
        x = 0.5 + radius * math.cos(angle)
        y = 0.5 + radius * math.sin(angle)
        positions.append((x, y))
        sizes.append(size_factor)

    # Randomize layout
    rotations = [random.randint(0, 359) for _ in symbols]
    combined = list(zip(positions, sizes))
    random.shuffle(combined)
    positions, sizes = zip(*combined)

    # Draw symbols
    for i, symbol in enumerate(symbols):
        rel_x, rel_y = positions[i]
        symbol_size = int(size * sizes[i])
        center_x = int(rel_x * size)
        center_y = int(rel_y * size)

        rotated = symbol.rotate(rotations[i], expand=True)
        resized = rotated.resize((symbol_size, symbol_size), Image.LANCZOS)

        paste_x = center_x - symbol_size // 2
        paste_y = center_y - symbol_size // 2

        mask = resized.split()[3] if resized.mode == "RGBA" else None
        card_image.paste(resized, (paste_x, paste_y), mask)

    return card_image


def play_game(game_id):
    st.title("Play Dobble!")

    # Session state initialization
    if "game_cards" not in st.session_state:
        st.session_state.game_cards = None
    if "pair_index" not in st.session_state:
        st.session_state.pair_index = 0
    if "card_pairs" not in st.session_state:
        st.session_state.card_pairs = []

    # Firebase setup
    try:
        storage, db = initialize_firebase()
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {str(e)}")
        return

    try:
        game_data = db.child("games").child(game_id).get().val()
        if not game_data:
            st.error(f"Game not found with ID: {game_id}")
            return

        st.write(f"### {game_data['title']}")
        st.write(f"Symbols per card: {game_data['symbols_per_card']}")
    except Exception as e:
        st.error(f"Error loading game data: {str(e)}")
        return

    # Generate cards once
    if st.session_state.game_cards is None:
        with st.spinner("Generating cards..."):
            cards, status = generate_game_cards(game_data)
            if not cards:
                st.error(f"Failed to generate cards: {status}")
                return
            st.session_state.game_cards = cards
            indices = list(range(len(cards)))
            st.session_state.card_pairs = list(itertools.combinations(indices, 2))
            random.shuffle(st.session_state.card_pairs)
            st.session_state.pair_index = 0

    # Handle out-of-bounds
    if st.session_state.pair_index >= len(st.session_state.card_pairs):
        st.warning("All pairs have been shown!")
        return

    card1_idx, card2_idx = st.session_state.card_pairs[st.session_state.pair_index]
    card1_symbols = st.session_state.game_cards[card1_idx]
    card2_symbols = st.session_state.game_cards[card2_idx]

    # Draw cards
    with st.spinner("Drawing cards..."):
        card1_img = draw_card_as_image(card1_symbols)
        card2_img = draw_card_as_image(card2_symbols)

    # Detect common symbol
    common_symbol = next(
        (s1 for s1 in card1_symbols for s2 in card2_symbols if similar_images(s1, s2)),
        None,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image(card1_img, use_column_width=True)
        st.markdown("#### Card 1")
    with col2:
        st.image(card2_img, use_column_width=True)
        st.markdown("#### Card 2")

    if st.button("Show Hint", key="show_hint"):
        if common_symbol:
            buf = io.BytesIO()
            common_symbol.save(buf, format="PNG")
            st.image(buf.getvalue(), width=100)
        else:
            st.warning("No common symbol found.")

    if st.button("Next Cards", key="next_cards"):
        st.session_state.pair_index += 1
        st.experimental_rerun()

    if st.button("Back to Main App", key="back_button"):
        st.session_state.clear()
        st.experimental_set_query_params()
        st.experimental_rerun()


# Streamlit app
def main():
    st.set_page_config(page_title="Dobble Card Generator", layout="centered")

    # Check for URL parameters - if game_id is present, go to play mode
    query_params = st.experimental_get_query_params()
    mode = query_params.get("mode", ["main"])[0]
    game_id = query_params.get("game_id", [None])[0]

    if mode == "play" and game_id:
        play_game(game_id)
        return

    # Check and install dependencies if needed
    try:
        check_and_install_dependencies()
    except Exception as e:
        st.error(f"Failed to install dependencies: {str(e)}")
        st.stop()

    # Initialize Firebase
    try:
        storage, db = initialize_firebase()
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {str(e)}")
        st.error(
            "Make sure you have set up Firebase Realtime Database and Storage correctly."
        )
        st.stop()

    # Create tabs
    tab1, tab2 = st.tabs(["Create New Game", "Saved Games"])

    with tab1:
        st.title("Dobble Card Generator")
        st.write("Create custom Dobble (Spot It) cards with your own images!")

        # Custom images section
        uploaded_images = []
        st.write("### Upload Your Images")
        st.write(
            "For the best results, upload square images with transparent backgrounds."
        )

        # Calculate how many images are needed
        symbols_per_card = st.selectbox(
            "Symbols Per Card",
            options=[3, 4, 6, 8, 12],
            index=2,  # Default to 6 symbols per card
        )

        order = symbols_per_card - 1
        total_symbols_needed = order**2 + order + 1

        st.write(f"You need to upload at least **{total_symbols_needed}** images.")

        # Create file uploader
        uploaded_files = st.file_uploader(
            "Upload your images",
            type=["png", "jpg", "jpeg", "gif", "bmp"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            # Process the images
            for file in uploaded_files:
                processed_img = preprocess_image(file)
                uploaded_images.append(processed_img)

            # Show how many images are uploaded
            if len(uploaded_images) < total_symbols_needed:
                st.warning(
                    f"You've uploaded {len(uploaded_images)} images. You need at least {total_symbols_needed} images."
                )
            else:
                st.success(
                    f"You've uploaded {len(uploaded_images)} images. That's enough to generate your cards!"
                )
            # Show the images in a grid
            cols = st.columns(5)  # 5 images per row
            for i, img in enumerate(uploaded_images[:15]):  # Show just the first 15
                with cols[i % 5]:
                    # Convert PIL Image to bytes for display
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.image(byte_im, width=100)

            if len(uploaded_images) > 15:
                st.write(f"... and {len(uploaded_images) - 15} more images")

        # Create a form for user input
        with st.form("dobble_form"):
            output_filename = st.text_input("Output Filename", value="dobble_cards.pdf")

            # Add a submit button
            submit_button = st.form_submit_button("Generate Cards")

        # Add a title field
        game_title = st.text_input(
            "Game Title",
            value=f"Dobble Game {datetime.datetime.now().strftime('%Y-%m-%d')}",
        )

        # Generate the cards when the user submits the form
        if submit_button:
            # Check if we have enough images
            if len(uploaded_images) < total_symbols_needed:
                st.error(
                    f"Not enough images! You need at least {total_symbols_needed} images, but you've only uploaded {len(uploaded_images)}."
                )
                return

            with st.spinner("Generating and saving your Dobble game..."):
                # Create a unique game ID
                game_id = str(uuid.uuid4())

                # Save images to Firebase and collect URLs
                image_urls = []
                for i, img in enumerate(uploaded_images):
                    try:
                        url = save_image_to_firebase(
                            storage, img, game_id, f"image_{i}"
                        )
                        image_urls.append(url)
                    except Exception as e:
                        st.error(f"Error saving image {i}: {str(e)}")
                        return

                # Create metadata for the game
                total_cards = symbols_per_card**2 - symbols_per_card + 1
                game_data = {
                    "id": game_id,
                    "title": game_title,
                    "created_at": datetime.datetime.now().isoformat(),
                    "symbols_per_card": symbols_per_card,
                    "total_symbols_needed": total_symbols_needed,
                    "total_cards": total_cards,
                    "image_urls": image_urls,
                }

                # Save game metadata to Firebase
                try:
                    save_game_to_firebase(db, game_id, game_data)
                    st.success(f"Game '{game_title}' saved successfully!")

                    # Create a temporary file for the PDF
                    temp_dir = tempfile.gettempdir()
                    output_path = os.path.join(temp_dir, output_filename)

                    # Generate the cards
                    generator = DobbleGenerator(
                        output_pdf=output_path,
                        symbols_per_card=symbols_per_card,
                        custom_images=uploaded_images,
                    )

                    success, message = generator.create_dobble_pdf()

                    if success:
                        # Create a download button for the PDF
                        download_button = get_binary_file_downloader_html(
                            output_path, output_filename
                        )
                        st.markdown(download_button, unsafe_allow_html=True)

                        # Show card info
                        st.info(f"Total cards in the deck: {total_cards}")
                    else:
                        st.error(message)

                except Exception as e:
                    st.error(f"Error saving game data: {str(e)}")

    with tab2:
        st.title("Saved Games")
        st.write("View and download your previously created Dobble games.")

        # Refresh button
        if st.button("Refresh Games List"):
            st.experimental_rerun()

        # Get all saved games
        try:
            saved_games = get_saved_games(db)

            if not saved_games:
                st.info("No saved games found. Create a new game first!")
            else:
                # Sort games by creation date (newest first)
                saved_games.sort(key=lambda x: x.get("created_at", ""), reverse=True)

                # Display games in a table
                st.write(f"Found {len(saved_games)} saved games:")

                for game in saved_games:
                    with st.expander(f"{game['title']} ({game['created_at'][:10]})"):
                        st.write(f"**Symbols per card:** {game['symbols_per_card']}")
                        st.write(f"**Total cards:** {game['total_cards']}")

                        # Display a few sample images if available
                        if "image_urls" in game and game["image_urls"]:
                            st.write("**Sample images:**")
                            cols = st.columns(5)
                            for i, url in enumerate(game["image_urls"][:5]):
                                with cols[i % 5]:
                                    st.image(url, width=60)

                            play_url = f"?mode=play&game_id={game['id']}"
                            st.markdown(
                                f"""
                                <a href="{play_url}" target="_self">
                                    <button style="background-color: #4CAF50; color: white; padding: 8px 16px; 
                                    border: none; border-radius: 4px; cursor: pointer; font-size: 16px;">
                                        Play Game
                                    </button>
                                </a>
                                """,
                                unsafe_allow_html=True,
                            )

                            if st.button(
                                f"Generate and Download PDF",
                                key=f"download_{game['id']}",
                            ):
                                with st.spinner("Generating PDF..."):
                                    pdf_path, pdf_filename = (
                                        generate_pdf_for_saved_game(storage, game)
                                    )
                                    if pdf_path:
                                        download_button = (
                                            get_binary_file_downloader_html(
                                                pdf_path, pdf_filename
                                            )
                                        )
                                        st.markdown(
                                            download_button, unsafe_allow_html=True
                                        )
                                    else:
                                        st.error("Failed to generate PDF")

                            # Delete game and images
                            if st.button(f"Delete Game", key=f"delete_{game['id']}"):
                                with st.spinner("Deleting game and images..."):
                                    success, msg = delete_game_from_firebase(
                                        storage,
                                        db,
                                        game["id"],
                                        game.get("image_urls", []),
                                    )
                                    if success:
                                        st.success(msg)
                                        st.experimental_rerun()
                                    else:
                                        st.error(msg)

        except Exception as e:
            st.error(f"Error loading saved games: {str(e)}")


if __name__ == "__main__":
    main()
