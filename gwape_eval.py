import streamlit as st
import numpy as np
import cv2 as cv
from dataclasses import dataclass
from typing import Any
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# Guarda log numa sheet do google
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from datetime import datetime
from pytz import timezone

# Supported colors in GRAPE RGB
####### Não esquecer converter em BGR #######

BLACK = (0, 0, 0)
GREY = (200, 200, 200)
WHITE = (255, 255, 255)

RED = (32, 49, 239)
ORANGE = (22, 114, 244)
YELLOW = (7, 199, 251)
LIGHT_GREEN = (44, 200, 115)
GREEN = (74, 138, 4)

SCORE_TO_COLOR = {
    5: GREEN,
    4: LIGHT_GREEN,
    3: YELLOW,
    2: ORANGE,
    1: RED,
}

DEFAULT_SCALING = {0: 1, 1: 0.9, 2: 1, 3: 1.1}

# OpenCV fonts map
FONTS = {
    "simplex": cv.FONT_HERSHEY_SIMPLEX,
    "plain": cv.FONT_HERSHEY_PLAIN,
    "duplex": cv.FONT_HERSHEY_DUPLEX,
    "complex": cv.FONT_HERSHEY_COMPLEX,
    "triplex": cv.FONT_HERSHEY_TRIPLEX,
    "complex_small": cv.FONT_HERSHEY_COMPLEX_SMALL,
    "script_simplex": cv.FONT_HERSHEY_SCRIPT_SIMPLEX,
    "script_complex": cv.FONT_HERSHEY_SCRIPT_COMPLEX,
    "italic": cv.FONT_ITALIC,
}

# Correspondance of GRAPE metric to berry in generated picture
METRIC_TO_BERRY_INDEX = {
    "type_of_measurements": (0, 0),
    "number_of_unitary_steps": (1, 0),
    "degree_of_automatization_and_miniaturization": (2, 0),
    "number_of_parameters_under_analysis": (3, 0),
    "sample_throughput": (0, 1),
    "energy_consumption": (1, 1),
    "hazard_classification_of_reagents": (2, 1),
    "sample_volume": (0, 2),
    "liquid_waste": (1, 2),
    "calibration_waste": (0, 3),
}

##########
CHOICES_TEST_1 = {
    5: "In-line/In-situ",
    4: "online/in situ",
    3: "On-site",
    2: "Ex-situ without storage",
    1: "Ex-situ with storage",
}
CHOICES_TEST_3 = {
    5: "Automatic-miniaturized",
    4: "Semi-automatic-miniaturized",
    3: "Non-automatic-miniaturized",
    3: "Automatic, non-miniaturized",
    2: "Semi-automatic, non-miniaturized",
    1: "Manual, non-miniaturized",
}
CHOICES_TEST_7 = {
    5: "No hazardous pictograms",
    3: "1 hazardous pictogram with indication warning",
    2: "2 hazardous pictograms with indication warning",
    1: "More than 3 hazardous pictograms with indication warning or at least 1 hazardous pictogram with indication danger",
}


# Geometry functions
def center_coordinates(index: tuple, R: int, a: float) -> np.array:
    """Compute a berry center given its index. The function computes the index assuming the top-left
    berry will be at (0, 0) so additional tranlation MUST BE APPLIED to place the berry in the proper
    place in the image.

    Args:
        index (tuple): The berry index as (x, z)
        R (int): Berry radius
        a (float): Berry overlap

    Returns:
        np.array: The (i, j)-th berry center coordinates.
    """
    # Extract x and y indices
    nx, ny = index

    # Horizontal coordinate
    cx = (2 - a) * R * (np.sin(np.pi / 6) * ny + nx)

    # Vertical coordinate
    cy = (2 - a) * R * np.cos(np.pi / 6) * ny

    return np.ceil(np.array([cx, cy])).astype(int)


def circe_with_border(
    image: np.ndarray,
    center: np.array,
    radius: float,
    color: tuple,
    thickness: float = 5.0,
    border_color: tuple = BLACK,
) -> np.array:
    """Draw a circle with a border on an image.

    Args:
        image (np.ndarray): Image to draw the circle on.
        center (np.array): Circle center.
        radius (float): Circle radius
        color (tuple): Color of the circle interior in BGR format.
        thickness (float, optional): The thickness of the border. Defaults to 5.0.
                                     note:(is NOT added to the radius).
        border_color (tuple, optional): The color of the border in BGR. Defaults to black ((0, 0, 0)).

    Returns:
        np.array: The updated image.
    """

    center_ = np.array(center).astype(int)
    radius_ = int(radius)

    # Border
    cv.circle(image, center_, radius_, border_color, thickness=thickness)

    # Inner circle
    cv.circle(image, center_, radius_, color, thickness=-1)

    return image


@dataclass
class GRAPESettings:

    # Image size
    size_x: int
    size_y: int

    # GRAPE settings
    radius: int
    overlap: float
    thickness: int

    # Bar settings
    length_ratio: float
    width_ratio: float
    overhang_ratio: float
    font: str

    # Berry scaling
    scaling: dict

    # Export target
    filename: str

    @classmethod
    def from_yaml(cls, filepath: str):
        """Creates an instance of the class from a settings contained in a YAML file.

        Args:
            filepath (str): Path to the settings file.

        Returns:
            GRAPESEttings: An instantiated settings class.
        """
        with open(filepath, "r") as file:
            settings = yaml.safe_load(file)

        # Arguments dictionary
        args = {}

        # Read image size
        args["size_x"] = settings["size"]["horizontal"]
        args["size_y"] = settings["size"]["vertical"]

        # Read geometry properties
        args.update(settings["geometry"])

        # Handle special case of radius
        if isinstance(args["radius"], int):
            pass
        elif isinstance(args["radius"], float):
            args["radius"] = int(
                args["radius"] * args["size_x"] / (3 * (2 - args["overlap"]) + 2)
            )
        else:
            raise ValueError(
                f'Invalid radius type {type(args["radius"])}. Accepted are: int, float'
            )

        # Read font
        args["font"] = FONTS[args["font"]]

        # Read berry scaling
        args["scaling"] = settings["scaling"]
        args["filename"] = settings["filename"]

        return cls(**args)  # Unpack the dictionary into the class


class GrapeImage:

    def __init__(self, radius: int, overlap: float, thickness: int = 5):
        """Initialize a grape image by describing the grape properties. This
        can be latered use to draw grapes on existing images.

        Args:
            radius (int): The berries radius
            overlap (float): The berries overlap
            thickness (int, optional): The berries border thickness. Defaults to 5.
        """

        # Create color list (default = all white)
        self.colors = {(i, j): WHITE for j in range(0, 4) for i in range(0, 4 - j)}
        self.weight = {(i, j): 1 for j in range(0, 4) for i in range(0, 4 - j)}

        # Store arguments internally
        self.radius = radius
        self.overlap = overlap
        self.thickness = thickness

        # Drawing the median bar defaults to False and can be adjusted by function
        self.draw_bar = False

    def read_scores(self, scores: dict, settings: dict) -> None:
        """Update grape image berry colors and radii based on a GRAPE scores.

        Args:
            scores (dict): _description_
        """
        # weighted score
        weight_sum = 0
        weighted_mean = 0

        for key, index in METRIC_TO_BERRY_INDEX.items():

            # Read score and value
            s, w = scores[key]

            # Update colors
            self.colors[index] = SCORE_TO_COLOR[s] if w != 0 else GREY

            # Read weight
            self.weight[index] = settings.scaling[w]

            # Compute weighted sum
            if w != 0:  # entries with value 0 don't contrinute to the mean
                weighted_mean += w * s
                weight_sum += w

        # Compute weighted average
        mean = weighted_mean / weight_sum
        mean = round(mean, 1)  # Round to one digit

        # Pick berry color
        bar_color = BLACK

        if mean >= 4.5:
            bar_color = GREEN
        elif mean < 4.5 and mean >= 3.5:
            bar_color = LIGHT_GREEN
        elif mean < 3.5 and mean >= 2.5:
            bar_color = YELLOW
        elif mean < 2.5 and mean >= 1.5:
            bar_color = ORANGE
        else:
            bar_color = RED

        # Set mean bar properties
        self.set_bar_properties(
            mean,
            bar_color,
            settings.length_ratio,
            settings.width_ratio,
            settings.overhang_ratio,
            settings.font,
        )

        return

    def set_bar_properties(
        self,
        mean: float,
        color: tuple,
        length_ratio: float = 0.4,
        width_ratio: float = 0.7,
        overhang: float = 0.5,
        bar_font: int = cv.FONT_HERSHEY_PLAIN,
    ) -> None:
        """Activate and specify mean-displaying bar or top of the grape.

        Args:
            mean (float): The mean to display.
            color (tuple): The bar color.
            length_ratio (float, optional): The bar length as a ratio*. Defaults to 0.4.
            width_ratio (float, optional): The bar width (as a precentage of the berry radius). Defaults to 0.7.
            overhang (float, optional): The bar distance from grape body (as a precentage of the berry radius). Defaults to 0.5.

            * For exapmle setting length_ratio to 0.4 means that each side of the bar will occupy 40% of the
              GRAPE width and leave 20% for the mean.

        """

        # Store mean and color
        self.bar_mean = mean
        self.bar_color = color

        # Store ratios
        self.bar_size_ratio = length_ratio
        self.bar_width = width_ratio * self.radius
        self.bar_overhang = overhang * self.radius

        # Display font
        self.font = bar_font

        # Activate bar display
        self.draw_bar = True

    def image_dim(self) -> tuple:
        """Compute the resulting GRAPE size in pixels.

        Returns:
            tuple: Tuple containing the GRAPE size as (horizontal pixels, verical pixels)
        """

        # Horizontal
        size_x = (8 - 3 * self.overlap) * self.radius

        # Vertical
        size_y = (
            3 * (2 - self.overlap) * np.cos(np.pi / 6) * self.radius + 2 * self.radius
        )

        # Add bar dimensions (if present)
        if self.draw_bar:
            size_y += self.bar_overhang + self.bar_width

        return (size_x, size_y)

    def draw(self, image: np.array, translation=np.zeros(2)) -> np.array:
        """Draw GRAPE on an input image starting from a given position.

        Args:
            image (np.array): The input image.
            translation (np.array): The coordinates where the shape will be drawn on the image.
                                    These correspond the top-left corner of the shape.
        """

        # Compute if there is an overhead due to drawing the median bar on top
        rectangle_overhead = np.array([self.radius, self.radius], dtype=float)

        if self.draw_bar:
            rectangle_overhead += np.array([0, self.bar_overhang + self.bar_width])

        # Draw berries
        for j in range(0, 4):
            for i in range(0, 4 - j):

                # Each berry corresponds to a circle with some properties
                image = circe_with_border(
                    image,
                    center_coordinates((i, j), self.radius, self.overlap)
                    + translation
                    + rectangle_overhead,
                    self.radius * self.weight[(i, j)],
                    self.colors[(i, j)],
                    thickness=self.thickness,
                    border_color=BLACK,
                )

        # Draw median bar (if exists)
        if self.draw_bar:

            # Get dimensions
            size_x, _ = self.image_dim()
            ratio = self.bar_size_ratio  # for more compact statements below

            # Left part of the median bar
            r1 = np.array([0.0, 0.0]) + translation
            r2 = np.array([size_x * ratio, self.bar_width]) + translation

            image = cv.rectangle(
                image, r1.astype(int), r2.astype(int), self.bar_color, thickness=-1
            )

            # Right part of the median bar
            r3 = np.array([(1 - ratio) * size_x, 0.0]) + translation
            r4 = np.array([size_x, self.bar_width]) + translation

            image = cv.rectangle(
                image, r3.astype(int), r4.astype(int), self.bar_color, thickness=-1
            )

            # Mean visualization
            # Note: 1.04 and 0.9 are visually chossen coefficients to place the
            #       median in (approximately) the middle of the bar
            location = (
                np.array([1.04 * ratio * size_x, self.bar_width * 0.9]) + translation
            )
            location = location.astype(int)

            font = self.font
            font_size = 0.025 * self.radius
            font_thickness = int(0.12 * self.radius)

            image = cv.putText(
                image,
                str(self.bar_mean),
                location,
                font,
                font_size,
                self.bar_color,
                font_thickness,
            )

        return image


def string_comparison(a: str, b: str) -> bool:
    """Compare two strings after eliminating redundant characters.

    Args:
        a (str): First string
        b (str): Second string

    Returns:
        bool: Whehter they are they same.
    """

    # (i) Convert to lower-case
    a = a.lower()
    b = b.lower()

    # (ii)  Eliminate error-pronce characters
    drop_characters = ["-", "/", "_", ","]

    for c in drop_characters:
        a = a.replace(c, "")
        b = b.replace(c, "")

    # (iii) Eliminate all whitespaces and
    a = a.replace(" ", "")
    b = b.replace(" ", "")

    return a == b


def retrieve_score(s: str, lookup_table: dict) -> Any:
    """Retrieve score from a lookup-table.

    Args:
        s (str): Input key
        test_keys (dict): Dictionary with test possible outcomes.

    Raises:
        KeyError: If the input key is not found in the look up table.

    Returns:
        Any: The value for the matched key from the lookup table.
    """

    for k, v in lookup_table.items():
        if string_comparison(s, k):
            return v

    raise KeyError(f"Unable to find key {k}. Available are: {lookup_table.keys()}")


# Define o estado inicial da variável de exibição
# Inicializar estados se não existirem
if "show_calibration_waste" not in st.session_state:
    st.session_state.show_calibration_waste = False

if "calibration" not in st.session_state:
    st.session_state.calibration = "No"

# -------------- SETTINGS --------------
PAGE_TITLE = "Green Analytical Chemistry"
PAGE_ICON = "♻"
PAGE_LAYOUT = "wide"
# --------------------------------------

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=PAGE_LAYOUT)

# --- HIDE STREAMLIT STYLE ---
# hide_st_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             header {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_st_style, unsafe_allow_html=True)

# --- INJECT CUSTUM CSS ---


def inject_custom_css():
    with open("styles/main.css") as f:
        st.markdown(f"<style> {f.read()}</style>", unsafe_allow_html=True)


inject_custom_css()

# --- LAYOUT CONFIG ---


st.title("♻" + PAGE_TITLE)

colL, colR = st.columns([4, 1])
colL.markdown("Evaluating the __greenness__ of analytical methods")
# colR.markdown(
#     ' <i class="fa-solid fa-link"></i>&nbsp;<a style="color: #5C6BC0; text-decoration: none;" href="https://www.alabe.pt" target="_blank">Sponsored by ALABE - Association of Enology Laboratories</a>', unsafe_allow_html = True
# )

colL.write(
    """
    ---

    Instructions: Enter the metrics (and weight) in the form sidebar and click __Evaluate__

    ---

    """
)

# Load settings file
settings_file = "./settings.yaml"
settings = GRAPESettings.from_yaml(settings_file)

# Establishing a Google Sheets connection
conn = st.connection("gsheets", type=GSheetsConnection)

# Fetch existing info
existing_data = conn.read(worksheet="Log", usecols=list(range(14)), ttl=5)
existing_data = existing_data.dropna(how="all")
# st.dataframe(existing_data)


def parse_GRAPE_metrics() -> dict:
    """Read GRAPE metrics from a loaded .csv file and extract GRAPE scores and weights.

    Returns:
        dict: Analysis resutls in format {key -> (value, weight)}
    """
    have_error = False
    weight_sum = 0
    results = {}  # key -> (score, weight)

    ################### Test 1: Promotion of in-situ measurements ###################
    key = "type_of_measurements"
    weight = type_of_measurements_w
    weight_sum += weight

    results[key] = (type_of_measurements, weight)

    ########## Test 2: Integration of analytical processes and operations ###########
    key = "number_of_unitary_steps"
    weight = number_of_unitary_steps_w
    weight_sum += weight

    i2 = number_of_unitary_steps

    results[key] = (max(6 - i2, 1), weight)

    ############ Test 3: Selection of automated and miniaturized methods ############
    key = "degree_of_automatization_and_miniaturization"
    weight = degree_of_automatization_and_miniaturization_w
    weight_sum += weight

    results[key] = (
        degree_of_automatization_and_miniaturization,
        weight,
    )

    ################## Test 4: Number of parameters under analysis ##################
    key = "number_of_parameters_under_analysis"
    weight = number_of_parameters_under_analysis_w
    weight_sum += weight

    i4 = number_of_parameters_under_analysis
    if type(i4) != int or i4 <= 0:
        colL.error('"number_of_parameters_under_analysis" must be a positive integer')
        have_error = True

    if i4 >= 5:
        results[key] = (5, weight)
    elif i4 >= 2 and i4 < 5:
        results[key] = (3, weight)
    else:
        results[key] = (1, weight)

    ########################### Test 5: Sample throughput ###########################
    key = "sample_throughput"
    weight = sample_throughput_w
    weight_sum += weight

    i5 = sample_throughput
    if i5 <= 0:
        colL.error('"sample_throughput" must be >= 0')
        have_error = True

    if i5 > 140:
        results[key] = (5, weight)
    elif i5 <= 140 and i5 > 120:
        results[key] = (4, weight)
    elif i5 <= 120 and i5 > 100:
        results[key] = (3, weight)
    elif i5 <= 100 and i5 > 80:
        results[key] = (2, weight)
    else:
        results[key] = (1, weight)

    ################## Test 6: Minimization of the waste of energy ##################
    key = "energy_consumption"
    weight = energy_consumption_w
    weight_sum += weight

    i6 = round(energy_consumption, 2)
    if i6 <= 0:
        colL.error('"energy_consumption" must be >= 0')
        have_error = True

    if i6 > 1.5:
        results[key] = (1, weight)
    elif i6 > 0.1 and i4 <= 1.5:
        results[key] = (3, weight)
    else:
        results[key] = (5, weight)

    ################## Test 7: Hazard classification of reagents ##################
    key = "hazard_classification_of_reagents"
    weight = hazard_classification_of_reagents_w
    weight_sum += weight

    results[key] = (
        hazard_classification_of_reagents,
        weight,
    )

    ############################ Test 8: Sample volume ############################
    key = "sample_volume"
    weight = sample_volume_w
    weight_sum += weight

    i8 = round(sample_volume, 2)

    if i8 < 0:
        colL.error('"sample_volume" must be >= 0')
        have_error = True

    if i8 < 0.1:
        results[key] = (5, weight)
    elif i8 >= 0.1 and i8 <= 1:
        results[key] = (3, weight)
    elif i8 > 1 and i8 <= 10:
        results[key] = (2, weight)
    else:
        results[key] = (1, weight)

    ############################ Test 9: Liquid waste #############################
    key = "liquid_waste"
    weight = consumable_material_waste_w
    weight_sum += weight

    if liquid_waste < 0:
        colL.error('"liquid_waste" must be >= 0')
        have_error = True

    if liquid_waste < 1:
        results[key] = (4, weight) if consumable_material_waste else (5, weight)
    elif liquid_waste >= 1 and liquid_waste < 10:
        results[key] = (2, weight) if consumable_material_waste else (3, weight)
    else:
        results[key] = (1, weight) if consumable_material_waste else (2, weight)

    ########################## Test 10: Calibration waste #########################
    key = "calibration_waste"
    weight = calibration_w
    weight_sum += weight

    # if calibration_waste < 0:
    #     colL.error('"calibration_waste" must be >= 0')
    #     have_error = True
    calibration = st.session_state.get("radio_10")
    calibration_waste = st.session_state.get("calibration_waste")

    if calibration is None or calibration == "No":
        results[key] = (5, weight)
    else:
        if calibration_waste > 10:
            results[key] = (1, weight)
        elif calibration_waste >= 1 and calibration_waste <= 10:
            results[key] = (2, weight)
        elif calibration_waste < 1 and calibration_waste > 0:
            results[key] = (3, weight)
        else:
            results[key] = (4, weight)

    print(weight_sum)
    if weight_sum == 0:
        colL.error("Sum of weights must be > 0")
        have_error = True

    # Ensure correct output format by
    # (i) ensuring scores are integers
    # (ii) convert weights to integers
    if have_error == True:
        return None
    else:
        for k, v in results.items():
            results[k] = (int(v[0]), int(np.round(v[1])))
        return results


def compute_scores():

    # Compute scores
    GRAPE_scores = parse_GRAPE_metrics()
    # st.write(GRAPE_scores)
    print(GRAPE_scores)

    if GRAPE_scores is not None:
        # Create base GRAPE object
        grape = GrapeImage(settings.radius, settings.overlap, settings.thickness)
        grape.read_scores(GRAPE_scores, settings)

        # Create output image
        SX, SY = (settings.size_x, settings.size_y)
        image = 255 * np.ones((SY, SX, 3), dtype=np.uint8)

        # Center image
        grape_size_x, grape_size_y = grape.image_dim()
        translation = np.round(
            np.array([(SX - grape_size_x), (SY - grape_size_y)]) / 2
        ).astype(int)

        # Draw GRAPE on image
        image = grape.draw(image, translation)

        # Convert the image from RGB to BGR
        bgr_img = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        # cv.imshow("GRAPE analysis", image)
        # cv.waitKey()

        # print(f"Succesfully exported image under {img_name}")
        colL.success("Analysis completed successfully")

        tab1, tab2 = colL.tabs(["Grape", "Metrics"])

        with tab1:
            st.image(bgr_img, "(Right click on the image and choose Save Image As...)")

        calibration = st.session_state.get("radio_10")
        calibration_waste = st.session_state.get("calibration_waste")

        new_data = pd.DataFrame(
            [
                {
                    "Metric": "Type of measurements",
                    "Value": CHOICES_TEST_1[type_of_measurements],
                    "Weight": type_of_measurements_w,
                },
                {
                    "Metric": "Number of unitary steps",
                    "Value": number_of_unitary_steps,
                    "Weight": number_of_unitary_steps_w,
                },
                {
                    "Metric": "Degree of automatization and miniaturization",
                    "Value": CHOICES_TEST_3[
                        degree_of_automatization_and_miniaturization
                    ],
                    "Weight": degree_of_automatization_and_miniaturization_w,
                },
                {
                    "Metric": "Number of parameters under analysis",
                    "Value": number_of_parameters_under_analysis,
                    "Weight": number_of_parameters_under_analysis_w,
                },
                {
                    "Metric": "Sample throughput",
                    "Value": sample_throughput,
                    "Weight": sample_throughput_w,
                },
                {
                    "Metric": "Energy consumption",
                    "Value": round(energy_consumption, 2),
                    "Weight": energy_consumption_w,
                },
                {
                    "Metric": "Hazard classification of reagents",
                    "Value": CHOICES_TEST_7[hazard_classification_of_reagents],
                    "Weight": hazard_classification_of_reagents_w,
                },
                {
                    "Metric": "Sample volume",
                    "Value": round(sample_volume, 2),
                    "Weight": sample_volume_w,
                },
                {
                    "Metric": "Consumable material waste",
                    "Value": consumable_material_waste
                    + (
                        f" (Liquid waste: {liquid_waste})"
                        if consumable_material_waste == "Yes"
                        else ""
                    ),
                    "Weight": consumable_material_waste_w,
                },
                {
                    "Metric": "Calibration waste",
                    "Value": calibration
                    + (
                        f" (Waste: {calibration_waste})" if calibration == "Yes" else ""
                    ),
                    "Weight": calibration_w,
                },
            ],
        )

        with tab2:
            st.dataframe(data=new_data, hide_index=True)

        # Create a new row of vendor data
        data_e_hora_atuais = datetime.now()
        fuso_horario = timezone("Europe/Lisbon")
        data_e_hora_lisboa = data_e_hora_atuais.astimezone(fuso_horario)
        data_e_hora_lisboa_str = data_e_hora_lisboa.strftime("%d/%m/%Y %H:%M:%S")

        new_data = pd.DataFrame(
            [
                {
                    "DataTime": data_e_hora_lisboa_str,
                    "Type of measurements": CHOICES_TEST_1[type_of_measurements],
                    "Number of unitary steps": number_of_unitary_steps,
                    "Degree of automatization and miniaturization": CHOICES_TEST_3[
                        degree_of_automatization_and_miniaturization
                    ],
                    "Number of parameters under analysis": number_of_parameters_under_analysis,
                    "Sample throughput": sample_throughput,
                    "Energy consumption": energy_consumption,
                    "Hazard classification of reagents": CHOICES_TEST_7[
                        hazard_classification_of_reagents
                    ],
                    "Sample volume": sample_volume,
                    "Consumable material waste": consumable_material_waste,
                    "Liquide waste": liquid_waste,
                    "Calibration": calibration,
                    "Calibration waste": calibration_waste,
                    "GRAPE scores": GRAPE_scores,
                }
            ]
        )

        # Add the new vendor data to the existing data
        updated_df = pd.concat([existing_data, new_data], ignore_index=True)
        # Update Google Sheets with the new vendor data
        conn.update(worksheet="Log", data=updated_df)


# --- FORM CONFIG ---
with st.sidebar.form(key="metrics_form"):
    st.header("Metrics:")

    # Test 1: Promotion of in-situ measurements
    with st.expander("Type of measurements"):
        type_of_measurements = st.selectbox(
            "Value:",
            CHOICES_TEST_1.keys(),
            format_func=lambda x: CHOICES_TEST_1[x],
            key="value_1",
        )
        type_of_measurements_w = st.number_input(
            "Weight:", min_value=0, max_value=3, step=1, key="weight_1"
        )
    # Test 2: Integration of analytical processes and operations
    with st.expander("Number of unitary steps"):
        number_of_unitary_steps = st.number_input("Value:", 1, None, 1, key="value_2")
        number_of_unitary_steps_w = st.number_input(
            "Weight:", min_value=0, max_value=3, step=1, key="weight_2"
        )
    # Test 3: Selection of automated and miniaturized methods
    with st.expander("Degree of automatization and miniaturization"):
        degree_of_automatization_and_miniaturization = st.selectbox(
            "Value:",
            CHOICES_TEST_3.keys(),
            format_func=lambda x: CHOICES_TEST_3[x],
            key="value_3",
        )
        degree_of_automatization_and_miniaturization_w = st.number_input(
            "Weight:", min_value=0, max_value=3, step=1, key="weight_3"
        )
    # Test 4: Choice of multi-analyte or multi-parameter methods
    with st.expander("Number of parameters under analysis"):
        number_of_parameters_under_analysis = st.number_input(
            "Value:", 1, None, 1, key="value_4"
        )
        number_of_parameters_under_analysis_w = st.number_input(
            "Weight:", min_value=0, max_value=3, step=1, key="weight_4"
        )
    # Test 5: Choose methods with a high sample throughput
    with st.expander("Sample throughput (number of samples per hour)"):
        sample_throughput = st.number_input("Value:", 1, None, 130, key="value_5")
        sample_throughput_w = st.number_input(
            "Weight:", min_value=0, max_value=3, step=1, key="weight_5"
        )
    # Test 6: Minimization of the waste of energy
    with st.expander("Energy consumption (KWh)"):
        energy_consumption = st.number_input(
            "Value:", 0.0, None, 0.01, 0.01, key="value_6"
        )
        energy_consumption_w = st.number_input(
            "Weight:", min_value=0, max_value=3, step=1, key="weight_6"
        )
    # Test 7: Use of safer reagents
    with st.expander("Hazard classification of reagents"):
        hazard_classification_of_reagents = st.selectbox(
            "Value:",
            CHOICES_TEST_7.keys(),
            format_func=lambda x: CHOICES_TEST_7[x],
            key="value_7",
        )
        hazard_classification_of_reagents_w = st.number_input(
            "Weight:", min_value=0, max_value=3, step=1, key="weight_7"
        )
    # Test 8: Minimization of sample size
    with st.expander("Sample volume (mL)"):
        sample_volume = st.number_input("Value:", 0.0, None, 0.3, 0.1, key="value_8")
        sample_volume_w = st.number_input(
            "Weight:", min_value=0, max_value=3, step=1, key="weight_8"
        )
    # Test 9: Minimize Waste
    with st.expander("Waste (mL)"):
        consumable_material_waste = st.radio(
            "Consumable material waste?", ("Yes", "No"), key="radio_9"
        )
        liquid_waste = st.number_input(
            "Waste(mL):",
            min_value=0.0,
            max_value=None,
            step=0.01,
            key="value_9",
        )
        consumable_material_waste_w = st.number_input(
            "Weight:",
            min_value=0,
            max_value=3,
            step=1,
            key="weight_9",
        )
    # Test 10: Calibration
    with st.expander("Calibration"):
        placeholder_calibration_radio = st.empty()

        placeholder_calibration_waste = st.empty()

        # calibration_waste = st.number_input(
        #     "Calibration liquid waste (mL):",
        #     min_value=0.0,
        #     max_value=None,
        #     step=0.01,
        #     key="value_10",
        # )
        calibration_w = st.number_input(
            "Weight:",
            min_value=0,
            max_value=3,
            step=1,
            key="weight_10",
        )

    # submit button
    submit_button = st.form_submit_button(label="Evaluate")

    # If the submit button is pressed
    if submit_button:
        compute_scores()

# Configurar o radio button no placeholder
with placeholder_calibration_radio:
    st.session_state.calibration = st.radio(
        "Calibration?",
        ("No", "Yes"),
        key="radio_10",
        on_change=lambda: st.session_state.update(
            {"show_calibration_waste": st.session_state.radio_10 == "Yes"}
        ),
    )

# Configurar o campo de input no placeholder
with placeholder_calibration_waste:
    if st.session_state.show_calibration_waste:
        st.session_state["calibration_waste"] = st.number_input(
            "Calibration liquid waste (mL):",
            min_value=0.0,
            max_value=None,
            step=0.01,
            key="value_10",
        )
    else:
        st.session_state["calibration_waste"] = None

# def main():
# if __name__ == "__main__":
#     main()
# streamlit run .\gwape_eval.py
###############################
