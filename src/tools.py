import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

@tool
def send_invoice(details: str) -> str:
    """
    Called when the customer confirms readiness to purchase.
    Args: 
    details: str (purchase details. Price, product name)
    """
    logger.info(f"Calling send_invoice with details: {details}")
    print(f"Calling send_invoice with details: {details}")


@tool
def handover_to_manager(lead_data: str) -> str:
    """
    Called when the customer needs to be transferred to a manager.
    Args:
    lead_data: str (brief conversation history)
    """
    logger.info(f"Calling handover_to_manager with data: {lead_data}")
    print(f"Calling handover_to_manager with data: {lead_data}")
