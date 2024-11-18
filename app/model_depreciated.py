from langchain.memory import ConversationSummaryMemory
from langchain.llms.base import BaseLanguageModel
from typing import Optional, List
import ollama

class OllamaModel(BaseLanguageModel):
    # Obligé de passer par cette classe pour que Langchain prenne en charge le forma du LLM, ici c'est "BaseLanguageModel" qui permet cette comptabilité.
    
    def __init__(self, model: str = "mistral:latest", ollama_options: dict = None):
        self.model = model
        self._ollama_option = ollama_options if ollama_options else {'temperature': 1}
        self.output = ""
        self.running = False

    def generate(self, prompt: str) -> str:
        # Cette méthode est nécessaire pour être compatible avec BaseLanguageModel d'après chatGPT **
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            stream=True,
            options=self._ollama_option
        )
        output = ""
        for chunk in response:
            output += chunk['response']
        return output
    
    def generate_prompt(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Génération de la réponse en tenant compte des éventuels stop tokens (Inutile dans notre cas mais encore une fois: needed for compatibility !!!!)
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            stream=True,
            options=self._ollama_option
        )
        output = ""
        for chunk in response:
            output += chunk['response']
            if stop and any(token in output for token in stop):
                break
        return output

    def predict(self, text: str) -> str:
        # Fournir une méthode de prédiction simple, pour pouvoir l'appeler dans les méthodes utilisées par LangChain. j'aurais pu utiliser un super je pense.. à voir
        return self.generate(text)

    def predict_messages(self, messages: List[str]) -> str:
        # pour résumer 
        context = "\n".join(messages)
        return self.generate(context)
    
    ## Necessaire pour héritage:
    
    def invoke(self, prompt: str) -> str:
        # équivaut à generate()
        return self.generate(prompt)

    ## les versions async
    async def apredict(self, text: str) -> str:
        return self.generate(text)
    
    async def agenerate(self, prompt: str) -> str:
        return self.generate(prompt)    
    
    async def apredict_messages(self, messages: list[dict[str, any]]) -> str:
        return self.predict_messages(messages)
    
    async def agenerate_prompt(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self.generate_prompt(prompt, stop)

# Utilisation de OllamaModel avec la mémoire
class ChatbotWithMemory:
    def __init__(self):
        self.model = OllamaModel(model="mistral:latest")  
        self.history = ConversationSummaryMemory(llm=self.model)

    def ans(self, input: str):
        context = self.history.load_memory_variables({}).get('history', "")

        prompt = (
            "Vous êtes un assistant intelligent. Utilisez les informations suivantes pour aider l'utilisateur.\n\n"
            "Mémoire du chatbot (à ne pas montrer à l'utilisateur) :\n"
            f"{context}\n\n"
            "Question de l'utilisateur :\n"
            f"{input}\n\n"
            "Répondez de manière claire et CONCISE et avec une mise en forme lisible et structurée :\n"
        )

        response = self.model.generate(prompt)

        # Mise à jour de la mémoire avec la question et la réponse
        self.history.save_context({"input": input}, {"output": response})

        return response

if __name__ == "__main__":
    chatbot = ChatbotWithMemory()
    response = chatbot.ans("Bonjour, qui es-tu ?")
    print(response)