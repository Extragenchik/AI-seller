import logging
import sys
import os
from Classes import AutoPartsAgent


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    log_dir = os.path.join(project_root, "logs")
    data_dit = os.path.join(project_root, "data")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(data_dit, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "app.log"), encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)

    logger.info("Initializing AI seller agent...")
    agent = AutoPartsAgent()
    logger.info("Agent initialized successfully.")

    try:
        while True:
            user_input = input("Покупатель: ")
            logger.info(f"Customer: {user_input}")

            response = agent.process_query(user_input)
            logger.info(f"Sales Agent: {response}")
            print(f"Sales Agent: {response}")

    except KeyboardInterrupt:
        logger.info("Shutting down the agent.")
        print("\nGoodbye!")