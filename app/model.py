from langchain.memory import ChatMessageHistory
import ollama

class OllamaModel:
    def __init__(self, model:str = "mistral:latest", ollama_options:dict = None):
        self.model = model
        self._ollama_option = ollama_options if ollama_options else {'temperature': 1}
        self.output = ""
        self.history = ChatMessageHistory()
        self.running = False
        
    def ans(self, input: str):
        prompt: str
        context: str
        self.running = True
        self.output = ""
        prompt = (
            "Vous êtes un assistant intelligent. Utilisez les informations suivantes pour aider l'utilisateur.\n\n"
            "Mémoire du chatbot (à ne pas montrer à l'utilisateur) :\n"
            f"{self.history.messages}\n\n"
            "Contexte pertinent :\n"
            f"{context}\n\n"
            "Question de l'utilisateur :\n"
            f"{input}\n\n"
            "Répondez de manière claire et CONCISE et avec une mise en forme lisible et structuré :\n"
        )
        # self.output = ollama.predict(self.model, prompt, **self._ollama_option)
        responce = ollama.generate(
            model = self.model,
            prompt = prompt,
            stream = True,
            options = self._ollama_option
            )
        
        self.output = ""
        for chunk in responce:
            self.output += chunk['responce']
            yield chunk['responce']
        
        self.history.add_user_message(input)
        self.history.add_ai_message(self.output)
        
        # return self.output
    
if __name__ == "__main__":
    model = OllamaModel()
    model.ans("Hello")