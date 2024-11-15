from model import OllamaModel

def main():
    bot = OllamaModel()
    while True:
        inp = input(">> ")
        if inp == "/quit":
            print('fin de conversation')
            break
        
        print("Bot:", end=" ")
        for chunk in bot.ans(inp):
            print(chunk, end='', flush=True)
        print()
        
    return 0

if __name__ == "__main__":
    main()