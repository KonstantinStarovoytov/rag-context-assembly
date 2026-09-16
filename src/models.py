from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceDocument:
    title: str
    content: str
    url: str
    vendor: str
    product: str
