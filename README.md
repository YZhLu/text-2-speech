# Classificador Automático de Notícias de Reconhecimento de Fala - PTBR

Este projeto representa a fase final de uma série de atividades realizadas para o componente curricular DCA0305 - PROJETO DE SISTEMAS BASEADOS EM APRENDIZADO DE MÁQUINA - T01 (2023.2 - 24M56) do Curso de Graduação em Engenharia da Computação da Universidade Federal do Rio Grande do Norte. O curso é ministrado pelo Professor Doutor Ivanovitch Medeiros Dantas da Silva, pertencente ao Departamento de Engenharia de Computação e Automação. A arquitetura do projeto consiste em dois modelos, Whisper e Bert, para realizar a classificação de notícias em português. O Bert foi submetido a um processo de ajuste fino (fine-tuning) utilizando a base de [notícias publicadas no Brasil](https://www.kaggle.com/datasets/diogocaliman/notcias-publicadas-no-brasil), enquanto as métricas de treinamento foram rastreadas usando o MLflow. Para integrar os dois modelos, foi desenvolvida uma interface gráfica utilizando o Gradio. O deploy do modelo está disponível no espaço do Hugging Face.

![Arquitetura](https://github.com/YZhLu/text-2-speech/blob/main/images/dgr.png)

## Conjunto de Dados:
O conjunto de dados consiste em 10 mil notícias divididas em 5 classes: esportes, economia, famosos, política e tecnologia.

Para mais informações, acesse o [link](https://www.kaggle.com/datasets/diogocaliman/notcias-publicadas-no-brasil).

## MLflow:
[MLflow](https://mlflow.org/) é uma plataforma de código aberto para o gerenciamento do ciclo de vida de projetos de aprendizado de máquina. Ela oferece um registro de experimentos, projetos, modelos e ferramentas de implantação. Sendo agnóstica em relação à linguagem e ao ambiente de execução, facilita o desenvolvimento, reprodução e implantação de soluções de ML.

## ZenML:
[ZenML](https://www.zenml.io/) é uma plataforma de código aberto para o gerenciamento de experimentos e pipelines em projetos de aprendizado de máquina. Ele simplifica o versionamento, organização e reprodução de experimentos, promovendo a colaboração entre equipes de ciência de dados.

## Gradio:
[Gradio](https://gradio.app/) é uma ferramenta de código aberto para criar interfaces de usuário simples para modelos de aprendizado de máquina. Facilita a criação de aplicativos interativos para visualização e interação com modelos ML, sem a necessidade de conhecimento em programação de interface.

## Pipeline de Dados:
![Pipeline](https://github.com/YZhLu/text-2-speech/blob/main/images/pipeline_data.png)

## Treino e Tracking:
Usando o MLflow, foram realizados 4 experimentos. Abaixo estão as métricas dos dois melhores experimentos capturadas com o MLflow (o modelo é avaliado por steps):

| Métricas  | Experimento 1  | Experimento 2  |
|:---------:|:--------------:|:--------------:|
| Acurácia  | 0.9649         | 0.9654         |
| F1        | 0.9674         | 0.9678         |
| Loss      | 0.0440         | 0.0451         |
| ROC AUC   | 0.9796         | 0.9795         |
| Épocas    | 2              | 1              |

### Acurácia x Steps:

<table>
  
  <tr>
   Experimento 1 <img src="https://github.com/YZhLu/text-2-speech/blob/main/images/acursxsteps1.png" alt="Experimento 1" width="800">  
  Experimento 2 <img src="https://github.com/YZhLu/text-2-speech/blob/main/images/acursxsteps2.png" alt="Experimento 2" width="800">
  </tr>

</table>

### Loss x Steps:

<table>
  
  <tr>
   Experimento 1 <img src="https://github.com/YZhLu/text-2-speech/blob/main/images/loss1.png" alt="Experimento 1" width="800">  
  Experimento 2 <img src="https://github.com/YZhLu/text-2-speech/blob/main/images/loss2.png" alt="Experimento 2" width="800">
  </tr>

</table>

### O modelo está disponível para inferência no [repositório do hugging face](https://huggingface.co/ClaudianoLeonardo/bert-finetuned_news_classifier-portuguese)


## Interface:
![Interface](https://github.com/YZhLu/text-2-speech/blob/main/images/Interface.png)

### O deploy do modelo está disponível no [Hugging Face Space](https://huggingface.co/spaces/gabrielblins/ASR_NewsClassifier_PTBR).

## Uso:
### Clone o repositório:
```md 
git clone https://github.com/YZhLu/text-2-speech
```
### Instale os requisitos:
```md
pip install -r requirements.txt
```
### Execute a função de treino:
```py
python train.py
```
### Para fine-tuning ou inferência, também é possível carregar o modelo diretamente do [repositório do Hugging Face](https://huggingface.co/ClaudianoLeonardo/bert-finetuned_news_classifier-portuguese).
