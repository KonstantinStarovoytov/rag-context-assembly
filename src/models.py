from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceDocument:
    title: str
    content: str
    url: str
    vendor: str
    product: str


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str

    content: str

    vendor: str
    product: str

    title: str
    heading_path: str
    source_url: str

    token_count: int

    @property
    def embedding_text(self) -> str:
        """
        Text that will actually be converted into an embedding.

        Adding structural context helps retrieval when chunk content
        alone is ambiguous.
        """
        return (
            f"Vendor: {self.vendor}\n"
            f"Product: {self.product}\n"
            f"Document: {self.title}\n"
            f"Section: {self.heading_path}\n\n"
            f"{self.content}"
        )
