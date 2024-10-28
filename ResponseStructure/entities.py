from pydantic import BaseModel, Field
from typing import List
from GraphMind_Logger import logger
class Entities(BaseModel):
    """
    Identify and capture information about entities from text
    """

    names: List[str] = Field(
        description="List of named entities extracted from the text, including people's names, company names, organization names, product names, brand names, and other proper nouns"
    )

logger.info("Entities model loaded")