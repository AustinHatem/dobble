import streamlit as st
import math
import random
import tempfile
import os
import io
import datetime
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
import base64

# Try to load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class DobbleGenerator:
    def __init__(
        self,
        output_pdf="dobble_cards.pdf",
        symbols_per_card=6,
        custom_images=None,
        card_background_color=(255, 255, 255),
    ):
        self.output_pdf = output_pdf
        # Fixed values for cards per row/column
        self.cards_per_row = 2
        self.cards_per_col = 2
        self.symbols_per_card = symbols_per_card
        self.custom_images = custom_images or []
        self.card_background_color = card_background_color  # RGB tuple

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

        # Calculate card size - use the smaller dimension to make squares
        page_margin = 1 * cm
        usable_width = width - (2 * page_margin)
        usable_height = height - (2 * page_margin)

        card_size = min(
            usable_width / self.cards_per_row, usable_height / self.cards_per_col
        )

        card_index = 0
        page_number = 1

        # Convert RGB tuple to ReportLab color components (0-1 scale)
        bg_r, bg_g, bg_b = [x / 255 for x in self.card_background_color]

        while card_index < len(cards):
            for row in range(self.cards_per_col):
                for col in range(self.cards_per_row):
                    if card_index >= len(cards):
                        break

                    card = cards[card_index]

                    # Calculate position for the card (centered in its cell)
                    cell_width = usable_width / self.cards_per_row
                    cell_height = usable_height / self.cards_per_col

                    # Center the square card in its cell
                    x = (
                        page_margin
                        + (col * cell_width)
                        + ((cell_width - card_size) / 2)
                    )
                    y = (
                        height
                        - page_margin
                        - ((row + 1) * cell_height)
                        + ((cell_height - card_size) / 2)
                    )

                    # Draw card background
                    c.setFillColorRGB(bg_r, bg_g, bg_b)
                    c.setStrokeColorRGB(0, 0, 0)  # Black border
                    c.rect(x, y, card_size, card_size, fill=1, stroke=1)

                    self._draw_card_symbols(
                        c,
                        card,
                        x,
                        y,
                        card_size,
                        card_size,
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
        size_factor = 0.25  # Slightly smaller for better spacing

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


# Streamlit app
def main():
    st.set_page_config(page_title="Dobble Card Generator", layout="centered")

    st.title("Dobble Card Generator")
    st.write("Create custom Dobble (Spot It!) cards with your own images!")

    # Custom images section
    uploaded_images = []
    st.write("### Upload Your Images")
    st.write("For the best results, upload square images with transparent backgrounds.")

    # Calculate how many images are needed
    symbols_per_card = st.selectbox(
        "Symbols Per Card",
        options=[3, 4, 6, 8, 12],
        index=2,  # Default to 6 symbols per card
    )

    # Card background color selection
    st.write("### Card Appearance")
    color_options = {
        "White": "#FFFFFF",
        "Light Blue": "#ADD8E6",
        "Light Green": "#90EE90",
        "Light Yellow": "#FFFFE0",
        "Light Pink": "#FFB6C1",
        "Light Gray": "#D3D3D3",
    }

    selected_color_name = st.selectbox(
        "Card Background Color", options=list(color_options.keys()), index=0
    )

    selected_color_hex = color_options[selected_color_name]

    # Show color preview
    st.markdown(
        f"""
        <div style="background-color: {selected_color_hex}; 
                    width: 100px; 
                    height: 100px; 
                    border-radius: 10px; 
                    border: 1px solid black;
                    display: flex;
                    align-items: center;
                    justify-content: center;">
            <span>Preview</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Convert hex color to RGB tuple
    selected_color_rgb = tuple(
        int(selected_color_hex.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)
    )

    order = symbols_per_card - 1
    total_symbols_needed = order**2 + order + 1
    total_cards = symbols_per_card**2 - symbols_per_card + 1

    st.write(f"You need to upload at least **{total_symbols_needed}** images.")
    st.write(f"This will generate a total of **{total_cards}** cards.")

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

    # Generate the cards when the user submits the form
    if submit_button:
        # Check if we have enough images
        if len(uploaded_images) < total_symbols_needed:
            st.error(
                f"Not enough images! You need at least {total_symbols_needed} images, but you've only uploaded {len(uploaded_images)}."
            )
            return

        with st.spinner("Generating cards..."):
            # Create a temporary file for the PDF
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, output_filename)

            # Generate the cards with the selected background color
            generator = DobbleGenerator(
                output_pdf=output_path,
                symbols_per_card=symbols_per_card,
                custom_images=uploaded_images,
                card_background_color=selected_color_rgb,
            )

            success, message = generator.create_dobble_pdf()

            if success:
                st.success("Cards generated successfully!")
                st.info(f"Total cards in the deck: {total_cards}")

                # Create a download button for the PDF
                download_button = get_binary_file_downloader_html(
                    output_path, output_filename
                )
                st.markdown(download_button, unsafe_allow_html=True)

                # Show some details about the generated cards
                st.write("### Card Details")
                st.write(f"- Each card has {symbols_per_card} symbols")
                st.write(f"- Total number of cards: {total_cards}")
                st.write("- Any two cards share exactly one symbol")
                st.write(f"- Background color: {selected_color_name}")
                st.write("- This is a mathematical property of projective planes")
            else:
                st.error(message)


if __name__ == "__main__":
    main()
