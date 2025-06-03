import csv
import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        logger.info("Initializing RAG system...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
        self.catalog_path = os.path.join(project_root, "data", "catalog.csv")
        self.script_path = os.path.join(project_root, "data", "sales_scripts.txt")

        logger.debug(f"Catalog path: {self.catalog_path}")
        logger.debug(f"Sales scripts path: {self.script_path}")

        self.vectorstore = self._build_vectorstore()
        logger.info("RAG system initialized successfully.")

    def _load_catalog(self):
        logger.info("Loading catalog data...")
        formatted_rows = []

        try:
            with open(self.catalog_path, "r", encoding="utf-8") as file:
                reader = csv.reader(file)
                headers = next(reader)

                for row in reader:
                    formatted_parts = [f"{headers[i]}: {row[i]}" for i in range(len(headers))]
                    formatted_row = ", ".join(formatted_parts)
                    formatted_rows.append(formatted_row)

            logger.info(f"Loaded {len(formatted_rows)} rows from catalog.")
            return formatted_rows

        except FileNotFoundError:
            logger.error(f"Catalog file not found: {self.catalog_path}")
        except Exception as e:
            logger.exception(f"Error reading catalog file: {e}")

        return []

    def _load_sales_script(self):
        logger.info("Loading sales script...")
        try:
            with open(self.script_path, "r", encoding="utf-8") as file:
                content = file.read()
            logger.info("Sales script loaded successfully.")
            return [content]

        except FileNotFoundError:
            logger.error(f"Sales script file not found: {self.script_path}")
        except Exception as e:
            logger.exception(f"Error reading sales script: {e}")

        return []

    def _build_vectorstore(self):
        logger.info("Building vector store...")
        try:
            catalog_data = self._load_catalog()
            script_data = self._load_sales_script()

            if not catalog_data and not script_data:
                logger.warning("No data loaded for vector store.")
                return None

            all_texts = catalog_data + script_data
            documents = self.text_splitter.create_documents(all_texts)
            vectorstore = FAISS.from_documents(documents, self.embeddings)
            logger.info("Vector store built successfully.")
            return vectorstore

        except Exception as e:
            logger.exception(f"Error building vector store: {e}")
            return None

    def search(self, query, k=10):
        logger.info(f"Searching for: '{query}'")
        if not self.vectorstore:
            logger.warning("Vector store is not initialized.")
            return []

        results = self.vectorstore.similarity_search(query, k=k)
        logger.info(f"Found {len(results)} results.")
        return results