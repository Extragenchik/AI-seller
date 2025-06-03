import logging
import os
from Classes import AutoPartsAgent


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    log_dir = os.path.join(project_root, "logs")
    data_dit = os.path.join(project_root, "data")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(data_dit, exist_ok=True)

    logger.info("Initializing agent for testing...")
    agent = AutoPartsAgent()
    logger.info("Agent initialized for testing.")

    # Пример диалога 1
    logger.info("Test case 1: Customer asks about washer motor for Golf 6")
    response = agent.process_query("Hello. Do you have a washer motor for Golf 6?")
    logger.info(f"Response: {response}")
    print(response)

    # Пример диалога 2
    logger.info("Test case 2: Customer asks about rear light for Passat B6")
    response = agent.process_query("How much does a rear light cost for Passat B6?")
    logger.info(f"Response: {response}")
    print(response)

    # Пример диалога 3
    logger.info("Test case 3: Customer asks about available parts")
    response = agent.process_query("What do you have?")
    logger.info(f"Response: {response}")
    print(response)