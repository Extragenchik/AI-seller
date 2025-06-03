import logging
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_ollama.chat_models import ChatOllama
from .RAGSystem import RAGSystem
from tools import send_invoice, handover_to_manager

logger = logging.getLogger(__name__)

class AutoPartsAgent:
    def __init__(self):
        logger.info("Initializing AutoPartsAgent...")
        self.rag = RAGSystem()
        self.llm = ChatOllama(model="qwen3:30b")
        self.prompt = self._create_prompt()
        self.tools = [send_invoice, handover_to_manager]
        logger.info("Creating tool calling agent...")
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent, tools=self.tools, verbose=True
        )
        self.history = ""
        logger.info("AutoPartsAgent initialized successfully.")

    def _create_prompt(self):
        logger.info("Creating prompt template...")
        template = """
        |
        "system": "!!! ВСЕ ОТВЕТЫ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ !!!\n\nТы - опытный продавец автозапчастей для немецких автомобилей. Твоя задача продавать запчасти для авто и вести диалог с клиентом",
        "instructions": [
        "1. Приветствовать клиента и уточнять его потребности в формате 'Добрый день! Какой запчастью вам интересуетесь?",
        "2. Использовать информацию из базы знаний "{context}" для поиска подходящих оригинальных запчастей и аналогов с указанием артикулов и цен, если нет подходящих данных, то их не используй и сообщи что ничего подходящего нет",
        "4. Обрабатывать возражения через сравнение характеристик, демонстрацию преимуществ",
        "5. Активно предлагать завершить сделку: 'Могу сформировать заказ прямо сейчас. Хотите оплатить онлайн или при получении?'",
        "6. При подтверждении заказа вызывать функцию send_invoice, при сложных вопросах - handover_to_manager",
        "7. Сохранять естественный диалоговый поток без шаблонных фраз"
        ],
        "tools": [send_invoice, handover_to_manager],
        "history": {history},
        |
        Покупатель: {query}
        "agent_scratchpad": {agent_scratchpad}
        Ответ продавца:
        """

        prompt = PromptTemplate(
            input_variables=["context", "history", "query", "agent_scratchpad"],
            template=template
        )
        logger.info("Prompt template created.")
        return prompt

    def process_query(self, query):
        logger.info(f"Processing query: {query}")
        context = ""
        if query:
            context = self.rag.search(query)

        input_data = {
            "context": context,
            "history": self.history,
            "query": query
        }

        try:
            result = self.agent_executor.invoke(input_data)
            logger.info(f"Agent response: {result['output']}")
            self.history += f"Customer: {query}\nSales Agent: {result['output']}\n"
            return result["output"]
        except Exception as e:
            logger.exception(f"Error during agent execution: {e}")
            return "An error occurred while processing your request."