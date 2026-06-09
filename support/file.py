import json
import os
import pickle
from typing import Any

from pathlib import Path

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging

    logger = logging.getLogger(__name__)


def load_file(path: str | Path) -> Any:
    if isinstance(path, Path):
        path = path.as_posix()
    extension = path.split(".")[-1]
    try:
        if extension == "txt":
            with open(path, "r", encoding="utf-8") as f:
                loaded_file = f.read()
        elif extension == "csv":
            import pandas as pd
            import unicodedata

            def to_nfc(val):
                if isinstance(val, str):
                    return unicodedata.normalize("NFC", val)
                elif isinstance(val, list):
                    return [to_nfc(x) for x in val]
                elif isinstance(val, dict):
                    return {to_nfc(k): to_nfc(v) for k, v in val.items()}
                return val

            df = pd.read_csv(path, encoding="utf-8-sig")
            df = df.map(to_nfc) if hasattr(df, "map") else df.applymap(to_nfc)
            df.columns = [
                to_nfc(col) if isinstance(col, str) else col for col in df.columns
            ]
            loaded_file = df
        elif extension == "json":
            with open(path, "r", encoding="utf-8-sig") as f:
                loaded_file = json.load(f)
        elif extension == "yaml":
            import yaml

            with open(path, "r", encoding="utf-8") as f:
                loaded_file = yaml.safe_load(f)
        elif extension == "pkl":
            with open(path, "rb") as f:
                loaded_file = pickle.load(f)
        else:
            raise ValueError(
                f"Unsupported file extension: {extension}",
            )
        logger.info(f"Loaded file from: {path}")
        return loaded_file
    except Exception as e:
        raise IOError(
            f"Error loading file: {e}",
        ) from e


def save_file(data: Any, path: str | Path):
    if isinstance(path, Path):
        path = path.as_posix()
    parent_dir = os.path.dirname(path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    sub_folder = os.path.basename(path)
    extension = sub_folder.split(".")[-1]
    try:
        if extension == "txt":
            with open(path, "w", encoding="utf-8-sig") as f:
                f.write(data)
        elif extension == "csv":
            import pandas as pd

            if isinstance(data, pd.DataFrame):
                import unicodedata

                def to_nfc(val):
                    if isinstance(val, str):
                        return unicodedata.normalize("NFC", val)
                    elif isinstance(val, list):
                        return [to_nfc(x) for x in val]
                    elif isinstance(val, dict):
                        return {to_nfc(k): to_nfc(v) for k, v in val.items()}
                    return val

                data = (
                    data.map(to_nfc) if hasattr(data, "map") else data.applymap(to_nfc)
                )
                data.columns = [
                    to_nfc(col) if isinstance(col, str) else col for col in data.columns
                ]
                data.to_csv(
                    path, index=False, encoding="utf-8-sig", lineterminator="\r\n"
                )
            else:
                raise ValueError(
                    "Data must be a pandas DataFrame for CSV format.",
                )
        elif extension == "json":
            import dataclasses
            class DataclassEncoder(json.JSONEncoder):
                def default(self, o):
                    if dataclasses.is_dataclass(o):
                        return dataclasses.asdict(o)
                    return super().default(o)
            with open(path, "w", encoding="utf-8-sig") as f:
                json.dump(data, f, ensure_ascii=False, indent=4, cls=DataclassEncoder)
        elif extension == "yaml":
            import yaml

            with open(path, "w", encoding="utf-8-sig") as f:
                yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        elif extension == "pkl":
            with open(path, "wb") as f:
                pickle.dump(data, f)
        elif extension in ["npy", "npz"]:
            import numpy as np

            np.save(path, data)
        else:
            raise ValueError(
                f"Unsupported file extension: {extension}",
            )
        logger.info(f"Saved file to: {path}")
    except Exception as e:
        logger.exception(f"Error saving file to {path}")
