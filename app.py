from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

import os
import google.generativeai as genai
from google.generativeai import GenerativeModel, GenerationConfig
from google.generativeai.types import HarmCategory, HarmBlockThreshold

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.header("Machine Learning & Generative Ai", divider="gray")

with st.sidebar:
     selected = option_menu(
          menu_title = "Main menu",
          options = ["Predict Disease", "Diet Preferences", "About"],
          icons = ["House", "Book", "Envelop"],
          default_index=0,
    )


def diseasePred():
    import pickle

    pickle = pickle.load(open("diseasePrediction.pkl","rb"))

# List of symptoms~
    symptoms = [
        "itching",
        "skin_rash",
        "nodal_skin_eruptions",
        "continuous_sneezing",
        "shivering",
        "chills",
        "joint_pain",
        "stomach_pain",
        "acidity",
        "ulcers_on_tongue",
        "muscle_wasting",
        "vomiting",
        "burning_micturition",
        "spotting_ urination",
        "fatigue",
        "weight_gain",
        "anxiety",
        "cold_hands_and_feets",
        "mood_swings",
        "weight_loss",
        "restlessness",
        "lethargy",
        "patches_in_throat",
        "irregular_sugar_level",
        "cough",
        "high_fever",
        "sunken_eyes",
        "breathlessness",
        "sweating",
        "dehydration",
        "indigestion",
        "headache",
        "yellowish_skin",
        "dark_urine",
        "nausea",
        "loss_of_appetite",
        "pain_behind_the_eyes",
        "back_pain",
        "constipation",
        "abdominal_pain",
        "diarrhoea",
        "mild_fever",
        "yellow_urine",
        "yellowing_of_eyes",
        "acute_liver_failure",
        "swelling_of_stomach",
        "swelled_lymph_nodes",
        "malaise",
        "blurred_and_distorted_vision",
        "phlegm",
        "throat_irritation",
        "redness_of_eyes",
        "sinus_pressure",
        "runny_nose",
        "congestion",
        "chest_pain",
        "weakness_in_limbs",
        "fast_heart_rate",
        "pain_during_bowel_movements",
        "pain_in_anal_region",
        "bloody_stool",
        "irritation_in_anus",
        "neck_pain",
        "dizziness",
        "cramps",
        "bruising",
        "obesity",
        "swollen_legs",
        "swollen_blood_vessels",
        "puffy_face_and_eyes",
        "enlarged_thyroid",
        "brittle_nails",
        "swollen_extremeties",
        "excessive_hunger",
        "extra_marital_contacts",
        "drying_and_tingling_lips",
        "slurred_speech",
        "knee_pain",
        "hip_joint_pain",
        "muscle_weakness",
        "stiff_neck",
        "swelling_joints",
        "movement_stiffness",
        "spinning_movements",
        "loss_of_balance",
        "unsteadiness",
        "weakness_of_one_body_side",
        "loss_of_smell",
        "bladder_discomfort",
        "foul_smell_of urine",
        "continuous_feel_of_urine",
        "passage_of_gases",
        "internal_itching",
        "toxic_look_(typhos)",
        "depression",
        "irritability",
        "muscle_pain",
        "altered_sensorium",
        "red_spots_over_body",
        "belly_pain",
        "abnormal_menstruation",
        "dischromic _patches",
        "watering_from_eyes",
        "increased_appetite",
        "polyuria",
        "family_history",
        "mucoid_sputum",
        "rusty_sputum",
        "lack_of_concentration",
        "visual_disturbances",
        "receiving_blood_transfusion",
        "receiving_unsterile_injections",
        "coma",
        "stomach_bleeding",
        "distention_of_abdomen",
        "history_of_alcohol_consumption",
        "fluid_overload.1",
        "blood_in_sputum",
        "prominent_veins_on_calf",
        "palpitations",
        "painful_walking",
        "pus_filled_pimples",
        "blackheads",
        "scurring",
        "skin_peeling",
        "silver_like_dusting",
        "small_dents_in_nails",
        "inflammatory_nails",
        "blister",
        "red_sore_around_nose",
        "yellow_crust_ooze",
    ]

    st.write("Using Machine Learning Model & Gemini Pro - Text only model")
    st.subheader("Disease Predictor & Diet Advisor")

    # Dictionary to store user responses (1 for Yes, 0 for No)
    responses = {}
    # Create columns for each symptom
    columns = st.columns(4) 

    # Loop through each symptom and ask for user input with unique key for each symptom
    for idx, symptom in enumerate(symptoms):
            with columns[idx % len(columns)]:
                responses[symptom] = st.radio(
                f"{symptom.replace('_', ' ')}",
                ('No', 'Yes'),
                key=f"response_{idx}"  # Unique key for each widget
            )
    # Convert 'Yes'/'No' responses to 1/0 for each symptom
    binary_responses = {symptom: 1 if response == 'Yes' else 0 for symptom, response in responses.items()}

    if st.button("Predict Disease"):
        if sum(binary_responses.values()) == 0:
            st.warning("Please select at least one symptom before predicting.")
        
        else:
            # Convert the responses to a DataFrame
            df_responses = pd.DataFrame([binary_responses])
            ans = pickle.predict(df_responses)
            # Display the DataFrame (optional, for debugging)
            # Display the prediction in bold
            st.markdown(f"**You have been diagnosed with: {str(ans[0]) if isinstance(ans, (list, np.ndarray)) else str(ans)}**")





def dietAdvisor():

    @st.cache_resource
    def loadmodel():
        model = genai.GenerativeModel("gemini-pro")
        return model


    def get_gemini_pro_text_response(
        model: GenerativeModel,
        contents: str,
        generation_config: GenerationConfig,
        stream: bool = True,):
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        generation_config = GenerationConfig(
            temperature=0.9,
            top_p=1.0,
            top_k=32,
            candidate_count=1,
            max_output_tokens=8192,
        )

        responses = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=stream,
        )

        final_response = []
        for response in responses:
            try:
                # st.write(response.text)
                final_response.append(response.text)
            except IndexError:
                # st.write(response)
                final_response.append("")
                continue
        return " ".join(final_response)
    

    
    st.write("Using Machine Learning Model & Gemini Pro - Text only model")
    st.subheader("Disease Predictor & Diet Advisor")
    text_model_pro = loadmodel()

    cuisine = st.selectbox(
        "What cuisine do you desire?",
        ("American", "Chinese", "French", "Indian", "Italian", "Japanese", "Mexican", "Turkish"),
        index=None,
        placeholder="Select your desired cuisine."
    )

    disease_name = st.selectbox(
        "What disease do you have?",
        ('Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
       'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
       'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
       'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
       'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
       'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
       'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
       'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
       'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
       'Osteoarthristis', 'Arthritis',
       '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
       'Urinary tract infection', 'Psoriasis', 'Impetigo'),
        index=None,
        placeholder="Select your disease."
    )

    dietary_preference = st.selectbox(
        "Do you have any dietary preferences?",
        ("Diabetese", "Glueten free", "Halal", "Keto", "Kosher", "Lactose Intolerance", "Vegan", "Vegetarian", "None"),
        index=None,
        placeholder="Select your desired dietary preference."
    )

    allergy = st.text_input(
        "Enter your food allergy:  \n\n", key="allergy", value="peanuts"
    )

    ingredient_1 = st.text_input(
        "Enter your first ingredient:  \n\n", key="ingredient_1", value="potatoes"
    )

    ingredient_2 = st.text_input(
        "Enter your second ingredient:  \n\n", key="ingredient_2", value="chicken"
    )

    ingredient_3 = st.text_input(
        "Enter your third ingredient:  \n\n", key="ingredient_3", value="latus"
    )

    # Task 2.5
    # Complete Streamlit framework code for the user interface, add the wine preference radio button to the interface.
    # https://docs.streamlit.io/library/api-reference/widgets/st.radio
    wine = st.radio(
        "Choose your type of wine:",
        ("Red", "White", "None")
    )

    max_output_tokens = 2048

    # Task 2.6
    # Modify this prompt with the custom chef prompt.
    prompt = f"""I am a chef tasked with creating {cuisine} recipes for a patient suffering from {disease_name}.\n
    The recipes should adhere to dietary restrictions specific to their condition, \n
    so avoid ingredients that may worsen the symptoms of {disease_name} and remember {dietary_preference} while creating recipes.\n
    However, don't include recipes that use ingredients with the patient's {allergy} allergy. \n
    I have {ingredient_1}, \n
    {ingredient_2}, \n
    and {ingredient_3} \n
    in my kitchen and other ingredients. \n
    The customer's wine preference is {wine} \n
    Please provide some for meal recommendations.
    For each recommendation include preparation instructions,
    time to prepare and the recipe title at the begining of the response.
    Then include the wine paring for each recommendation.
    At the end of the recommendation provide the calories associated with the meal
    and the nutritional facts."""

    config = {
        "temperature": 0.8,
        "max_output_tokens": 2048,
    }

    generate_t2t = st.button("Generate my recipes.", key="generate_t2t")
    if generate_t2t and prompt:
        # st.write(prompt)
        with st.spinner("Generating your recipes using Gemini..."):
            first_tab1, first_tab2 = st.tabs(["Recipes", "Prompt"])
            with first_tab1:
                response = get_gemini_pro_text_response(
                    text_model_pro,
                    prompt,
                    generation_config=config,
                )
                if response:
                    st.write("Your recipes:")
                    st.write(response)
            with first_tab2:
                st.text(prompt)

if selected == "Predict Disease":
        diseasePred()

if selected == "Diet Preferences":
        dietAdvisor()

if selected == "About":
        st.markdown("About Disease Prediction Model and Diet preferances Model using Generative AI")
        st.write("This project uses Machine Learning to predict the disease that user is suffering based on the health conditions. Also If the user want dietry advices in order to take precautions, the Gen-Ai model can give recipies based on incredients the user has.")