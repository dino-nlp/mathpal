import opik
from config import settings
from langchain_openai import ChatOpenAI
from opik.integrations.langchain import OpikTracer

import core.logger_utils as logger_utils
from core import lib
from core.db.documents import GradeDocument
from core.rag.prompt_templates import SelfQueryTemplate

logger = logger_utils.get_logger(__name__)


class SelfQuery:
    
    @staticmethod
    @opik.track(name="SelQuery.generate_response")
    def generate_response(query: str) -> str | None:
        prompt = SelfQueryTemplate().create_template()
        model = ChatOpenAI(
            model=settings.OPENAI_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
        chain = prompt | model
        
        # Fix OpikTracer usage - use RunnableConfig with callbacks
        from langchain_core.runnables import RunnableConfig  
        opik_tracer = OpikTracer(tags=["SelfQuery"])
        config = RunnableConfig(callbacks=[opik_tracer])
        response = chain.invoke({"question": query}, config=config)
        response = response.content
        grade_name = response.strip("\n ")

        if grade_name == "none":
            return None

        logger.info(
            f"Successfully extracted the user full name from the query.",
            grade_name=grade_name,
        )
        
        grade_id = GradeDocument.get_or_create(name=grade_name)

        return grade_id
