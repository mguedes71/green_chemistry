import streamlit as st
import numpy as np
import cv2 as cv

# Guarda log numa sheet do google
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from datetime import datetime
from pytz import timezone


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

# from PIL import Image, ImageFilter

centers = {
    0: (100, 100),
    1: (168, 100),
    2: (236, 100),
    3: (304, 100),
    4: (134, 160),
    5: (202, 160),
    6: (270, 160),
    7: (168, 220),
    8: (236, 220),
    9: (202, 280),
}

score_to_color_rgb = {
    5: (0, 128, 0),
    4: (0, 204, 73),
    3: (0, 255, 210),
    2: (0, 156, 210),
    1: (0, 51, 190),
}

score_to_color_bgr = {
    5: (0, 128, 0),
    4: (73, 204, 0),
    3: (210, 255, 0),
    2: (210, 156, 0),
    1: (190, 51, 0),
}


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

    Instructions: Enter the metrics in the form sidebar and click __Evaluate__

    ---

    """
)

# Establishing a Google Sheets connection
conn = st.connection("gsheets", type=GSheetsConnection)

# Fetch existing info
existing_data = conn.read(worksheet="Log", usecols=list(range(14)), ttl=5)
existing_data = existing_data.dropna(how="all")
# st.dataframe(existing_data)


def draw_berry(img, center, radius, color, thickness=5, border_color=(0, 0, 0)):
    cv.circle(img, center, radius, border_color, thickness=thickness)
    cv.circle(img, center, radius, color, thickness=-1)

    return img


def parse_GRAPE_metrics():
    # Prepare output structure
    GRAPE_scores = 10 * [1]

    # Test 1: Promotion of in-situ measurements
    GRAPE_scores[0] = type_of_measurements

    # Test 2: Integration of analytical processes and operations
    i2 = number_of_unitary_steps

    GRAPE_scores[1] = max(6 - i2, 1)

    # Test 3: Selection of automated and miniaturized methods
    GRAPE_scores[2] = degree_of_automatization_and_miniaturization

    # Test 4: Choice of multi-analyte or multi-parameter methods
    i4 = number_of_parameters_under_analysis

    if type(i4) != int or i4 <= 0:
        colL.error('"number_of_parameters_under_analysis" must be a positive integer')

    if i4 >= 5:
        GRAPE_scores[3] = 5
    elif i4 >= 2 and i4 < 5:
        GRAPE_scores[3] = 3
    else:
        GRAPE_scores[3] = 1

    # Test 5: Choose methods with a high sample throughput
    i5 = sample_throughput

    if i5 <= 0:
        colL.error('"sample_throughput" must be >= 0')

    if i5 > 140:
        GRAPE_scores[4] = 5
    elif i5 <= 140 and i5 > 120:
        GRAPE_scores[4] = 4
    elif i5 <= 120 and i5 > 100:
        GRAPE_scores[4] = 3
    elif i5 <= 100 and i5 > 80:
        GRAPE_scores[4] = 2
    else:
        GRAPE_scores[4] = 1

    # Test 6: Minimization of the waste of energy
    i6 = energy_consumption

    if i6 <= 0:
        colL.error('"energy_consumption" must be >= 0')

    if i6 > 1.5:
        GRAPE_scores[5] = 1
    elif i6 > 0.1 and i4 <= 1.5:
        GRAPE_scores[5] = 3
    else:
        GRAPE_scores[5] = 5

    # Test 7: Use of safer reagents
    GRAPE_scores[6] = hazard_classification_of_reagents

    # Test 8: Minimization of sample size
    i8 = sample_volume

    if i8 <= 0:
        colL.error('"sample_volume" must be >= 0')

    if i8 < 0.1:
        GRAPE_scores[7] = 5
    elif i8 >= 0.1 and i8 <= 1:
        GRAPE_scores[7] = 3
    elif i8 >= 2 and i8 <= 10:
        GRAPE_scores[7] = 2
    else:
        GRAPE_scores[7] = 1

    # Test 9: Minimize Waste
    if liquid_waste < 0:
        colL.error('"liquid_waste" must be >= 0')

    if liquid_waste < 1:
        GRAPE_scores[8] = 4 if consumable_material_waste else 5
    elif liquid_waste >= 1 and liquid_waste < 10:
        GRAPE_scores[8] = 2 if consumable_material_waste else 3
    else:
        GRAPE_scores[8] = 1 if consumable_material_waste else 2

    # Test 10: Calibration
    if calibration_waste < 0:
        colL.error('"calibration_waste" must be >= 0')

    if not calibration:
        GRAPE_scores[9] = 5
    else:
        if calibration_waste > 10:
            GRAPE_scores[9] = 1
        elif calibration_waste >= 1 and calibration_waste <= 10:
            GRAPE_scores[9] = 2
        elif calibration_waste < 1 and calibration_waste > 0:
            GRAPE_scores[9] = 3
        else:
            GRAPE_scores[9] = 4

    return GRAPE_scores


def compute_scores():
    # Define empty canvas
    size = 400
    img = 255 * np.ones((size, size, 3), dtype=np.uint8)

    # Compute scores
    GRAPE_scores = parse_GRAPE_metrics()
    # st.write(GRAPE_scores)

    # 1. Initialize image
    img = 255 * np.ones((size, size, 3), dtype=np.uint8)

    # 2. Draw berries
    for i in range(10):
        color = score_to_color_bgr[GRAPE_scores[i]]
        img = draw_berry(img, centers[i], 30, color)

    # Save result
    # img_name = score_file["filename"]
    # cv.imwrite(img_name, img)
    # print(f"Succesfully exported image under {img_name}")
    colL.success("Analysis completed successfully")
    colL.image(img, "(Right click on the image and choose Save Image As...)")

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
    type_of_measurements = st.selectbox(
        "Type of measurements:",
        CHOICES_TEST_1.keys(),
        format_func=lambda x: CHOICES_TEST_1[x],
    )
    # Test 2: Integration of analytical processes and operations
    number_of_unitary_steps = st.number_input("Number of unitary steps:", 1, None, 1)
    # Test 3: Selection of automated and miniaturized methods
    degree_of_automatization_and_miniaturization = st.selectbox(
        "Degree of automatization and miniaturization:",
        CHOICES_TEST_3.keys(),
        format_func=lambda x: CHOICES_TEST_3[x],
    )
    # Test 4: Choice of multi-analyte or multi-parameter methods
    number_of_parameters_under_analysis = st.number_input(
        "Number of parameters under analysis:", 1, None, 1
    )
    # Test 5: Choose methods with a high sample throughput
    sample_throughput = st.number_input("Sample throughput:", 1, None, 130)
    # Test 6: Minimization of the waste of energy
    energy_consumption = st.number_input("Energy consumption:", 0.0, None, 0.01, 0.01)
    # Test 7: Use of safer reagents
    hazard_classification_of_reagents = st.selectbox(
        "Hazard classification of reagents:",
        CHOICES_TEST_7.keys(),
        format_func=lambda x: CHOICES_TEST_7[x],
    )
    # Test 8: Minimization of sample size
    sample_volume = st.number_input("Sample volume:", 0.0, None, 0.3, 0.1)
    # Test 9: Minimize Waste
    consumable_material_waste = st.radio("Consumable material waste:", ("Yes", "No"))
    liquid_waste = st.number_input("Liquid waste:", 0, None, 12, 1)
    # Test 10: Calibration
    calibration = st.radio("Calibration:", ("Yes", "No"))
    calibration_waste = st.number_input("Calibration waste:", 0, None, 0)
    # submit button
    submit_button = st.form_submit_button(label="Evaluate")

    # If the submit button is pressed
    if submit_button:
        compute_scores()

# def main():
# if __name__ == "__main__":
#     main()
