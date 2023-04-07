from pathlib import Path

from llama_index import GPTSimpleVectorIndex, ServiceContext, download_loader
from llama_index.indices.base import BaseGPTIndex


class PDFDocument:  # pylint: disable=too-few-public-methods
    """
    A class that represents a PDF document.
    """

    def __init__(self, pdf_path: str):
        """
        PDFDocument class constructor.

        :param pdf_path: The path to the PDF file to use.
        """
        self.document = download_loader("PDFReader")().load_data(file=Path(pdf_path))

    def create_index(self, service_context: ServiceContext) -> BaseGPTIndex:
        """
        Create an index of the document. For more information, see the following links:
        https://gpt-index.readthedocs.io/en/latest/reference/indices/vector_store.html#gpt_index.indices.vector_store.vector_indices.GPTSimpleVectorIndex

        :param service_context: The context used to create the index. For more information, see the following links:
        https://gpt-index.readthedocs.io/en/latest/reference/service_context.html
        :return:
        """
        return GPTSimpleVectorIndex.from_documents(
            self.document, service_context=service_context
        )
