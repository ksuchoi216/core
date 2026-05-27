from langfuse import Langfuse
from dataclasses import dataclass
from loguru import logger


def create_dataset(name: str, description: str, metadata: dict):
    langfuse = Langfuse()
    dataset = langfuse.create_dataset(
        name=name,
        description=description,
        metadata=metadata,
    )
    logger.info("Dataset created: {}", dataset)
    return dataset


def create_dataset_item(
    dataset_name: str,
    input_text: str,
    expected_output_text: str,
    metadata: dict | None = None,
):
    langfuse = Langfuse()
    langfuse.create_dataset_item(
        dataset_name=dataset_name,
        input={"text": input_text},
        expected_output={"text": expected_output_text},
        metadata=metadata,
    )
    logger.info("Dataset item created: {}", input_text)


@dataclass
class DatasetItem:
    input: dict
    expected_output: dict
    metadata: dict | None = None


def create_dataset_items(dataset_name: str, dataset_items: list[dict]):
    langfuse = Langfuse()
    # check dataset_items with DatasetItem.
    for dataset_item in dataset_items:
        if not isinstance(dataset_item, DatasetItem):
            raise ValueError(f"Invalid dataset item: {dataset_item}")
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input=dataset_item.input,
            expected_output=dataset_item.expected_output,
            metadata=dataset_item.metadata,
        )

    logger.info("Dataset items created: {}", len(dataset_items))
