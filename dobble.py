import streamlit as st
import math
import random
import tempfile
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
import base64


class DobbleGenerator:
    def __init__(
        self,
        output_pdf="dobble_cards.pdf",
        cards_per_row=2,
        cards_per_col=2,
        symbols_per_card=6,
    ):
        self.output_pdf = output_pdf
        self.cards_per_row = cards_per_row
        self.cards_per_col = cards_per_col
        self.symbols_per_card = symbols_per_card

        self.order = self.symbols_per_card - 1
        self.total_symbols_needed = self.order**2 + self.order + 1

        self.shapes = [
            "circle",
            "square",
            "triangle",
            "star",
            "heart",
            "diamond",
            "pentagon",
            "hexagon",
        ]

        self.colors = [
            colors.red,
            colors.blue,
            colors.green,
            colors.orange,
            colors.purple,
            colors.yellow,
            colors.pink,
            colors.brown,
            colors.black,
            colors.gray,
        ]

        # Generate exactly N visually unique shape-color combinations
        self.symbols = []
        used_combinations = set()
        for shape in self.shapes:
            for color in self.colors:
                if (shape, color) not in used_combinations:
                    self.symbols.append((shape, color))
                    used_combinations.add((shape, color))
                if len(self.symbols) == self.total_symbols_needed:
                    break
            if len(self.symbols) == self.total_symbols_needed:
                break

        if len(self.symbols) < self.total_symbols_needed:
            raise ValueError(
                "Not enough unique shape-color combinations to generate a valid Dobble deck."
            )

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
        # Define appropriate positions based on number of symbols per card
        if self.symbols_per_card == 6:
            positions = [
                (0.25, 0.8),
                (0.75, 0.8),
                (0.15, 0.5),
                (0.5, 0.5),
                (0.85, 0.5),
                (0.5, 0.2),
            ]
            sizes = [0.15, 0.15, 0.15, 0.2, 0.15, 0.15]
        elif self.symbols_per_card == 8:
            positions = [
                (0.25, 0.8),
                (0.75, 0.8),
                (0.15, 0.6),
                (0.5, 0.65),
                (0.85, 0.6),
                (0.2, 0.3),
                (0.5, 0.25),
                (0.8, 0.3),
            ]
            sizes = [0.12, 0.12, 0.12, 0.15, 0.12, 0.12, 0.15, 0.12]
        else:
            # For any other number of symbols, generate positions in a circle
            positions = []
            center_size = 0.2  # Size of center symbol
            outer_sizes = [0.15] * (self.symbols_per_card - 1)  # Size of outer symbols

            # Add center position
            positions.append((0.5, 0.5))
            sizes = [center_size]

            # Add positions in a circle
            radius = 0.3
            for i in range(self.symbols_per_card - 1):
                angle = 2 * math.pi * i / (self.symbols_per_card - 1)
                x_pos = 0.5 + radius * math.cos(angle)
                y_pos = 0.5 + radius * math.sin(angle)
                positions.append((x_pos, y_pos))
                sizes.extend(outer_sizes)

        # Cut positions if there are more than symbols
        positions = positions[: len(symbols)]
        sizes = sizes[: len(symbols)]

        rotations = [random.randint(0, 359) for _ in range(len(symbols))]

        # Randomize symbol positions
        combined = list(zip(positions, sizes))
        random.shuffle(combined)
        positions, sizes = zip(*combined)

        for i, (shape, color) in enumerate(symbols):
            if i >= len(positions):
                break
            rel_x, rel_y = positions[i]
            size_factor = sizes[i]
            symbol_size = min(width, height) * size_factor
            center_x = x + rel_x * width
            center_y = y + rel_y * height

            self._draw_shape(
                canvas, shape, color, center_x, center_y, symbol_size, rotations[i]
            )

    def _draw_shape(self, canvas, shape, color, x, y, size, rotation=0):
        canvas.saveState()
        canvas.setFillColor(color)
        canvas.setStrokeColor(colors.black)
        canvas.setLineWidth(1)
        canvas.translate(x, y)
        if shape != "circle":
            canvas.rotate(rotation)
        half_size = size / 2

        if shape == "circle":
            canvas.circle(0, 0, half_size, fill=1)
        elif shape == "square":
            canvas.rect(-half_size, -half_size, size, size, fill=1)
        elif shape == "triangle":
            height = size * math.sqrt(3) / 2
            points = [
                (-half_size, -height / 3),
                (half_size, -height / 3),
                (0, height * 2 / 3),
            ]
            self._draw_polygon(canvas, points, fill=1)
        elif shape == "star":
            points = []
            outer_radius = half_size
            inner_radius = half_size * 0.4
            for i in range(10):
                radius = outer_radius if i % 2 == 0 else inner_radius
                angle = math.pi / 2 + 2 * math.pi * i / 10
                points.append((radius * math.cos(angle), radius * math.sin(angle)))
            self._draw_polygon(canvas, points, fill=1)
        elif shape == "heart":
            r = half_size * 0.6
            canvas.circle(-r, r, r, fill=1)
            canvas.circle(r, r, r, fill=1)
            points = [(-2 * r, r), (0, -2.5 * r), (2 * r, r)]
            self._draw_polygon(canvas, points, fill=1)
        elif shape == "diamond":
            points = [(0, half_size), (half_size, 0), (0, -half_size), (-half_size, 0)]
            self._draw_polygon(canvas, points, fill=1)
        elif shape == "pentagon":
            points = []
            for i in range(5):
                angle = math.pi / 2 + 2 * math.pi * i / 5
                points.append(
                    (half_size * math.cos(angle), half_size * math.sin(angle))
                )
            self._draw_polygon(canvas, points, fill=1)
        elif shape == "hexagon":
            points = []
            for i in range(6):
                angle = 2 * math.pi * i / 6
                points.append(
                    (half_size * math.cos(angle), half_size * math.sin(angle))
                )
            self._draw_polygon(canvas, points, fill=1)
        canvas.restoreState()

    def _draw_polygon(self, canvas, points, fill=1):
        path = canvas.beginPath()
        path.moveTo(points[0][0], points[0][1])
        for x, y in points[1:]:
            path.lineTo(x, y)
        path.close()
        canvas.drawPath(path, fill=fill, stroke=1)


# Function to create a download link for the PDF
def get_pdf_download_link(pdf_path, filename):
    """Generate a link to download the pdf file"""
    with open(pdf_path, "rb") as file:
        pdf_contents = file.read()

    b64_pdf = base64.b64encode(pdf_contents).decode()
    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}">Download the PDF</a>'

    return href


# Streamlit app
def main():
    st.set_page_config(page_title="Dobble Card Generator", layout="centered")

    st.title("Dobble Card Generator")
    st.write("Create custom Dobble (Spot It) cards to print yourself!")

    # Create a form for user input
    with st.form("dobble_form"):
        col1, col2 = st.columns(2)

        with col1:
            cards_per_row = st.number_input(
                "Cards Per Row", min_value=1, max_value=4, value=2, step=1
            )

        with col2:
            cards_per_col = st.number_input(
                "Cards Per Column", min_value=1, max_value=4, value=2, step=1
            )

        symbols_per_card = st.selectbox(
            "Symbols Per Card",
            options=[3, 4, 6, 8, 12],
            index=2,  # Default to 6 symbols per card
        )

        output_filename = st.text_input("Output Filename", value="dobble_cards.pdf")

        # Add a submit button
        submit_button = st.form_submit_button("Generate Cards")

    # Generate the cards when the user submits the form
    if submit_button:
        with st.spinner("Generating Dobble cards..."):
            # Create a temporary file for the PDF
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, output_filename)

            try:
                # Generate the cards
                generator = DobbleGenerator(
                    output_pdf=output_path,
                    cards_per_row=cards_per_row,
                    cards_per_col=cards_per_col,
                    symbols_per_card=symbols_per_card,
                )

                success, message = generator.create_dobble_pdf()

                if success:
                    st.success(message)

                    # Create a download link for the PDF
                    download_link = get_pdf_download_link(output_path, output_filename)
                    st.markdown(download_link, unsafe_allow_html=True)

                    # Show card info
                    total_cards = symbols_per_card**2 - symbols_per_card + 1
                    st.info(f"Total cards in the deck: {total_cards}")

                    # Show some tips
                    with st.expander("Printing Tips"):
                        st.write(
                            """
                        - Print on cardstock for durability
                        - Cut out the cards carefully
                        - For best results, laminate the cards after cutting
                        - For a more portable game, print on smaller paper or reduce scale
                        """
                        )
                else:
                    st.error(message)

            except Exception as e:
                st.error(f"Error generating cards: {str(e)}")


if __name__ == "__main__":
    main()
