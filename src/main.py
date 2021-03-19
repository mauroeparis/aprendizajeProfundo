import logging
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from .constants import (
    LANGUAGE,
    PRETRAINED_EMBEDDINGS_PATH,
    TEST_DATA_PATH,
    TOKEN_TO_INDEX_PATH,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
)
from .hypterparameters import (
    EPOCHS,
    # MLPClassifier
    DROPOUT,
    EMBEDDINGS_SIZE,
    FREEZE_EMBEDINGS,
    HIDDEN_LAYERS,
    # DataLoader
    BATCH_SIZE,
    # MeliChallengeDataset
    RANDOM_BUFFER_SIZE,
    # Optimizer
    LR,
    WEIGHT_DECAY,
)
from .dataset import MeliChallengeDataset
from .mlp import MLPClassifier
from .utils import PadSequences


logging.basicConfig(
    format="%(asctime)s: %(levelname)s - %(message)s",
    level=logging.INFO
)


if __name__ == "__main__":
    pad_sequences = PadSequences(
        pad_value=0,
        max_length=None,
        min_length=1
    )

    logging.info("Building training dataset")
    train_dataset = MeliChallengeDataset(
        dataset_path=TRAIN_DATA_PATH,
        random_buffer_size=RANDOM_BUFFER_SIZE,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=pad_sequences,
        drop_last=False,
    )

    if VALIDATION_DATA_PATH:
        logging.info("Building validation dataset")

        validation_dataset = MeliChallengeDataset(
            dataset_path=VALIDATION_DATA_PATH,
            random_buffer_size=1,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=pad_sequences,
            drop_last=False,
        )
    else:
        validation_dataset = None
        validation_loader = None

    if TEST_DATA_PATH:
        logging.info("Building test dataset")

        test_dataset = MeliChallengeDataset(
            dataset_path=TEST_DATA_PATH,
            random_buffer_size=1
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=pad_sequences,
            drop_last=False
        )
    else:
        test_dataset = None
        test_loader = None

    mlflow.set_experiment(f"diplodatos.{LANGUAGE}")

    with mlflow.start_run():
        logging.info("Starting experiment")

        # Log all relevent hyperparameters
        mlflow.log_params({
            "model_type": "Multilayer Perceptron",
            "embeddings": PRETRAINED_EMBEDDINGS_PATH,
            "hidden_layers": HIDDEN_LAYERS,
            "dropout": DROPOUT,
            "embeddings_size": EMBEDDINGS_SIZE,
            "epochs": EPOCHS
        })

        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        logging.info("Building classifier")
        model = MLPClassifier(
            pretrained_embeddings_path=PRETRAINED_EMBEDDINGS_PATH,
            token_to_index=TOKEN_TO_INDEX_PATH,
            n_labels=train_dataset.n_labels,
            hidden_layers=HIDDEN_LAYERS,
            dropout=DROPOUT,
            vector_size=EMBEDDINGS_SIZE,
            freeze_embedings=FREEZE_EMBEDINGS,
        )

        model = model.to(device)
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        logging.info("Training classifier")
        for epoch in trange(EPOCHS):
            model.train()
            running_loss = []

            for idx, batch in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                data = batch["data"].to(device)
                target = batch["target"].to(device)
                output = model(data)
                loss_value = loss(output, target)
                loss_value.backward()
                optimizer.step()
                running_loss.append(loss_value.item())

            mlflow.log_metric(
                "train_loss",
                sum(running_loss) / len(running_loss),
                epoch,
            )

            if validation_dataset:
                logging.info("Evaluating model on validation")
                model.eval()
                running_loss = []
                targets = []
                predictions = []

                with torch.no_grad():
                    for batch in tqdm(validation_loader):
                        data = batch["data"].to(device)
                        target = batch["target"].to(device)

                        output = model(data)
                        running_loss.append(
                            loss(output, target).item()
                        )
                        targets.extend(batch["target"].numpy())
                        predictions.extend(
                            output.argmax(axis=1).detach().cpu().numpy())

                    mlflow.log_metric(
                        "validation_loss",
                        sum(running_loss) / len(running_loss),
                        epoch,
                    )
                    mlflow.log_metric(
                        "validation_bacc",
                        balanced_accuracy_score(targets, predictions),
                        epoch,
                    )

        if test_dataset:
            logging.info("Evaluating model on test")
            model.eval()
            running_loss = []
            targets = []
            predictions = []

            with torch.no_grad():
                for batch in tqdm(test_loader):
                    data = batch["data"].to(device)
                    target = batch["target"].to(device)
                    output = model(data)
                    running_loss.append(
                        loss(output, target).item()
                    )
                    targets.extend(batch["target"].numpy())
                    predictions.extend(
                        output.argmax(axis=1).detach().cpu().numpy())

                mlflow.log_metric(
                    "test_loss",
                    sum(running_loss) / len(running_loss),
                    epoch,
                )
                mlflow.log_metric(
                    "test_bacc",
                    balanced_accuracy_score(targets, predictions),
                    epoch,
                )

