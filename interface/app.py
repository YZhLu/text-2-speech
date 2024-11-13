import gradio as gr
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

model_id = 'ClaudianoLeonardo/bert-finetuned_news_classifier-portuguese'
tokenizer_classifier = AutoTokenizer.from_pretrained(model_id)
model_classifier = AutoModelForSequenceClassification.from_pretrained(model_id)

model_id2 = "Stopwolf/distil-whisper-large-v2-pt"

# Carregar modelos do Hugging Face
whisper_model = pipeline('automatic-speech-recognition', model = model_id2)

text_classification_model = AutoModelForSequenceClassification.from_pretrained(model_id)
text_classification_tokenizer = AutoTokenizer.from_pretrained(model_id)

id2label = {0:'economia', 1:'esportes', 2:'famosos', 3:'politica', 4:'tecnologia'}

def get_text(logits):
  sigmoid = torch.nn.Sigmoid()
  probs = sigmoid(logits.squeeze().cpu())
  predictions = np.zeros(probs.shape)
  predictions[np.where(probs >= 0.5)] = 1
  predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
  return predicted_labels[0]

# Função para realizar a inferência
def inference(audio):
    # Realizar inferência no modelo Whisper para reconhecimento de fala
    # Obter texto da saída do modelo Whisper
    try:
      sr, y = audio
    except:
       return "Erro ao carregar o áudio ou insira um áudio válido"
    
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    transcribed_text = whisper_model({"sampling_rate": sr, "raw": y})["text"]

    # Realizar inferência no modelo de classificação de texto
    text_input = text_classification_tokenizer(transcribed_text, return_tensors="pt", padding=True)
    text_output = text_classification_model(**text_input)
    # Obter a classe predita
    predicted_class = get_text(text_output["logits"])

    return f"Texto transcrito: {transcribed_text}\nClasse predita: {predicted_class}"

# Criar interface gráfica com Gradio
iface = gr.Interface(
    fn=inference,
    inputs=gr.Audio(),
    outputs="text",
    live=True
)

# Executar a interface
iface.launch(debug=True)