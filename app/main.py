import langflow
import rag_application  # Importieren Sie Ihr RAG-Anwendungsmodul

def main():
    # Initialisieren Sie Langflow LLM
    langflow.initialize()

    # Starten Sie Ihre RAG-Anwendung
    rag_application.run()

if __name__ == "__main__":
    main()