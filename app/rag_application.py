import json

def run():
    # Laden Sie den exportierten Flow aus der JSON-Datei
    with open('flow.json', 'r') as file:
        exported_flow = json.load(file)
    
    # Initialisieren Sie den Flow (angenommen, Sie haben eine Flow-Klasse, die dies unterstützt)
    flow = Flow(exported_flow)
    
    # Ausführen des Flows
    flow.run()

    print("RAG-Anwendung läuft...")