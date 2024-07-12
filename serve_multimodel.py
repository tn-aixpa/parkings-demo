
from darts.models import NBEATSModel
from zipfile import ZipFile

def init(context):
    # Qua ti setti l'id del modello che vuoi caricare
    model_id = "8be239dc-8798-48ff-bc3b-80d1d80cc2af"

    # prendi l'entity model sulla base dell'id
    model = context.project.get_model(entity_id=model_id)
    path = model.download()
    local_path_model = "extracted_model/"
    # Qua fai unzip immagino
    with ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(local_path_model)
    
    # codice che carica il modello
    input_chunk_length = 24
    output_chunk_length = 12
    name_model_local = local_path_model +"parcheggi_predictor_model.pt"
    mm = NBEATSModel(
            input_chunk_length,
            output_chunk_length
    ).load(name_model_local)

    # settare model nel context di nuclio (non su project che è il context nostro)
    context.setattr("model", mm)
    #setattr(context, "model", mm)

def serve(context, event):

    # Sostanzialmente invochiamo la funzione con una chiamata REST
    # Nel body della richiesta mandi l'inference input
    
    if isinstance(event.body, bytes):
        body = json.loads(event.body)
    context.logger.info(f"Received event: {body}")
    inference_input = body["inference_input"]
    
    return context.model.predict(n=output_chunk_length*2,
                                 series=inference_input)
