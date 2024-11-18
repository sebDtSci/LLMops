from langchain_ollama.llms import OllamaLLM
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.llms.base import BaseLanguageModel


class OllamaModel:
    def __init__(self, model:str = "mistral:latest", ollama_options:dict = None):
        self.ollama_model = OllamaLLM(
            model=model,
            options=ollama_options if ollama_options else {'temperature': 1} # vérifier le paramétrage de la température 0 pour déterministe langchain et 1 pour ollama_python
            )
        self.output = ""
        self.history = ConversationSummaryMemory(llm=self.ollama_model)
        self.running = False
        
    def ans(self, input: str):
        prompt: str
        context: str
        self.running = True
        self.output = ""
        
        mem = self.history.load_memory_variables({}).get('history', "")
        
        prompt = (
            "Vous êtes un assistant intelligent. Utilisez les informations suivantes pour aider l'utilisateur.\n\n"
            "Mémoire du chatbot (à ne pas montrer à l'utilisateur) :\n"
            f"{mem}\n\n"
            # "Contexte pertinent :\n"
            # f"{context}\n\n"
            "Question de l'utilisateur :\n"
            f"{input}\n\n"
            "Répondez de manière claire et CONCISE et avec une mise en forme lisible et structuré :\n"
        )
        
        response = self.ollama_model.generate(
            prompts=[prompt],
            stream=True
        )
        
        
        # self.output = ""
        # for chunk in response:
            # self.output += chunk['response']
            # yield chunk['response']
        
        for chunk in response:
            if isinstance(chunk, tuple) and chunk[0] == 'generations':
                generation_list = chunk[1]
                if generation_list and isinstance(generation_list[0], list):
                    generation_chunk = generation_list[0][0] 
                    if hasattr(generation_chunk, 'text'):
                        self.output += generation_chunk.text
                        yield generation_chunk.text
            
        


        
        self.history.save_context({"input": input}, {"output": self.output})
        
        return 0
    
if __name__ == "__main__":
    model = OllamaModel(model="mistral:latest")
    model.ans("Hello")