from langchain import OpenAI
from llama_index import GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from llama_index.indices.base import BaseGPTIndex

from pdf_gpt.pdf_document import PDFDocument


class PDFBot:
    """
    A bot class for generating PDF-based answers.
    """

    def __init__(self, index: BaseGPTIndex):
        """
         A class that represents a PDF bot.
        :param index: The base index to use. For more information,
        see: https://gpt-index.readthedocs.io/en/latest/reference/indices.html
        """
        self.index = index

    @classmethod
    def from_index_file(cls, path: str) -> "PDFBot":
        """
        Creates an PDFBot object from an index file. Currently only GPTSimpleVectorIndex is supported.

        :param path: The path to the index file.
        :return: PDFBot object.
        """
        return cls(GPTSimpleVectorIndex.load_from_disk(path))

    @classmethod
    def from_pdf_file(
        cls,
        path: str,
        llm: OpenAI = OpenAI(model_name="text-davinci-003"),
        prompt_helper: PromptHelper = PromptHelper(
            max_input_size=4097, num_output=256, max_chunk_overlap=20
        ),
    ) -> "PDFBot":
        """
        Creates an PDFBot object from an pdf file.

        :param path: The path to the pdf file.
        :param llm: An OpenAI LLM object. The default value is text-davinci-003.
        :param prompt_helper: The PromptHelper object. For more information,
        see: https://gpt-index.readthedocs.io/en/latest/reference/service_context/prompt_helper.html
        :return: PDFBot object.
        """
        return cls(
            PDFDocument(pdf_path=path).create_index(
                ServiceContext.from_defaults(
                    llm_predictor=LLMPredictor(llm), prompt_helper=prompt_helper
                )
            )
        )

    def ask(
        self, query: str, mode: str = "default", response_mode: str = "default"
    ) -> str:
        """
        Ask questions through a bot.

        :param query: Question string.
        :param mode: An index can have a variety of query modes. For instance, you can choose to specify mode="default"
        or mode="embedding" for a list index.
        mode="default" will a create and refine an answer sequentially through the nodes of the list.
        mode="embedding" will synthesize an answer by fetching the top-k nodes by embedding similarity.
        :param response_mode:
        - default: For the given index, “create and refine” an answer by sequentially going through each Node;
        make a separate LLM call per Node. Good for more detailed answers.
        - compact: For the given index, “compact” the prompt during each LLM call by stuffing as many Node text chunks
        that can fit within the maximum prompt size. If there are too many chunks to stuff in one prompt,
        “create and refine” an answer by going through multiple prompts.
        - tree_summarize: Given a set of Nodes and the query, recursively construct a tree and return the root node as
        the response. Good for summarization purposes.
        :return: The answer string to the question.
        """
        return str(self.index.query(query, mode=mode, response_mode=response_mode))
