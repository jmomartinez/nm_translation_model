from dataclasses import dataclass


@dataclass
class ModelArgs:
    optimizer: str
    loss_function: str
    epochs: int
    batch_size: int
    in_vocab_size: int
    out_vocab_size: int
    max_in_length: int
    max_out_length: int
    validation_split: float = None
    vector_space_size: int = 32
