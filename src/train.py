from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import mlflow
from utils import compute_metrics, tokenize_function
from fetch_data import fetch_data
from pre_processing import pre_process_data
from segregation import segregate_data

class NewsClassifierTrainer:
    """Class for training a news classifier using transformers and mlflow."""

    def __init__(self, model_id, num_labels, batch_size, metric_name, model_name, train_dataset, val_dataset, num_train_epochs):
        """
        Initializes the NewsClassifierTrainer.

        Args:
            model_id (str): Identifier for the pre-trained model.
            num_labels (int): Number of labels for the classification task.
            batch_size (int): Batch size for training.
            metric_name (str): Name of the evaluation metric.
            model_name (str): Name for the fine-tuned model.
            train_dataset (Dataset): Training dataset.
            val_dataset (Dataset): Validation dataset.
            num_train_epochs (int): Number of training epochs.
        """
        self.model_id = model_id
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.metric_name = metric_name
        self.model_name = model_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_train_epochs = num_train_epochs
        self.encoded_train = None
        self.encoded_val = None
        self.id2label = {0: 'economia', 1: 'esportes', 2: 'famosos', 3: 'politica', 4: 'tecnologia'}
        self.label2id = {label: idx for idx, label in self.id2label.items()}
        self.args = None
        self.trainer = None

    def _set_args(self):
        """
        Sets the TrainingArguments for the Trainer.
        """
        args = TrainingArguments(
            output_dir=self.model_name,
            evaluation_strategy="steps",
            save_strategy="steps",
            warmup_steps=5,
            logging_steps=100,
            save_steps=500,
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_train_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model=self.metric_name
        )
        self.args = args

    def _set_trainer(self):
        """
        Sets the Trainer for training the model.
        """
        self._set_args()
        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.encoded_train,
            eval_dataset=self.encoded_val,
            compute_metrics=compute_metrics
        )

    def _train_and_tracking(self):
        """
        Trains the model and logs metrics with mlflow.
        """
        self._set_trainer()
        with mlflow.start_run():
            self.trainer.train()
            mlflow.log_params(self.args.to_dict())
            final_metrics = self.trainer.evaluate(self.encoded_val)
            for metric_name, metric_value in final_metrics.items():
                mlflow.log_metric(f"final_{metric_name}", metric_value)
        mlflow.end_run()

    def train_and_tracking(self):
        """
        Trains the model and logs metrics with mlflow.
        """
        self.encoded_train = self.train_dataset.map(tokenize_function, batched=True)
        self.encoded_val = self.val_dataset.map(tokenize_function, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(self.model_id, num_labels=self.num_labels, id2label=self.id2label,label2id=self.label2id)
        self.model = model

        self._train_and_tracking()

if __name__ == "__main__":
    raw_data = fetch_data('data\Historico_de_materias.csv')
    data_train = pre_process_data(raw_data)
    train_dataset, val_dataset = segregate_data(data_train)

    news_classifier_trainer = NewsClassifierTrainer(
        model_id="bert-base-uncased",
        num_labels=5,
        batch_size=8,
        metric_name='f1',
        model_name="bert-finetuned_news_classifier-portuguese",
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_train_epochs=2
    )
    news_classifier_trainer.train_and_tracking()
