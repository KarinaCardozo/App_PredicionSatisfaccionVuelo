import gradio as gr
import pandas as pd
import pickle


# Define params names
PARAMS_NAME = [
    "Age",
    "Class",
    "Wifi",
    "Booking",
    "Seat",
    "Checkin",
]

# Load model
with open(r"bcamp\Semana 7\Satisfaccion_del_vuelo\model\rf.pkl", "rb") as f:
    model = pickle.load(f)

# Columnas
COLUMNS_PATH = r"bcamp\Semana 7\Satisfaccion_del_vuelo\model\categories_ohe.pickle"
with open(COLUMNS_PATH, 'rb') as handle:
    ohe_tr = pickle.load(handle)


def predict(*args):
    answer_dict = {}

    for i in range(len(PARAMS_NAME)):
        answer_dict[PARAMS_NAME[i]] = [args[i]]

    single_instance = pd.DataFrame.from_dict(answer_dict)
    
    # Reformat columns
    single_instance_ohe = pd.get_dummies(single_instance).reindex(columns = ohe_tr).fillna(0)
    
    prediction = model.predict(single_instance_ohe)

    response = format(prediction[0], '.2f')

    return response


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Customer Satisfaction
        """
    )

    with gr.Row():
        with gr.Column():

            gr.Markdown(
                """
                ## Predecir si el cliente va a estar satisfecho
                """
            )

            Age = gr.Slider(label="Edad", minimum=0, maximum=90, step=1, randomize=True)

            Class = gr.Radio(
                label="Clase",
                choices=["Business", "Eco", "Eco Plus"],
                value="Business"
                )
            
            Wifi = gr.Slider(label="Servicio Wifi", minimum=0, maximum=5, step=1, randomize=True)

            Booking = gr.Slider(label="Facilidad de registro", minimum=0, maximum=5, step=1, value=5)

            Seat = gr.Dropdown(
                label="Comodidad del asiento",
                choices=[0, 1, 2, 3, 4, 5],
                multiselect=False,
                value=5,
                )
            
            Checkin = gr.Dropdown(
                label="Experiencia con el Checkin",
                choices=[0, 1, 2, 3, 4, 5],
                multiselect=False,
                value=5,
                )

            
        with gr.Column():

            gr.Markdown(
                """
                ## Predicción
                """
            )

            label = gr.Label(label="Score")
            predict_btn = gr.Button(value="Evaluar")
            predict_btn.click(
                predict,
                inputs=[
                Age,
                Class,
                Wifi,
                Booking,
                Seat,
                Checkin,
                ],
                outputs=[label],
            )

    gr.Markdown(
        """
        <p style='text-align: center'> 
                '_blank'>Proyecto Kari 🤗
            </a>
        </p>
        """
    )

demo.launch()
