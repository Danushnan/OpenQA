import os
from settings import settings
from mistralai.client import Mistral
from tracking import log_tokens

class LLM:
    def __init__(self, model="ministral-3b-latest"):
        self.model = model
        
    def response(self, message: str, question_id: str):
        with Mistral(api_key=settings.MISTRAL_API_KEY) as mistral:
            response = mistral.chat.complete(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": message
                    }
                ]
            )
        log_tokens(model=self.model, response=response)
        return response.choices[0].message.content
    