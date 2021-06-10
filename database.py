import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import json

cred = credentials.Certificate(
    "charlotapp-firebase-adminsdk-y8n29-630f0f23a9.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

# read data
# Getting a Document with a known ID

intents = {"intents": []}
detect_duplicate_by_tag = []

with open('intents.json') as json_file:
    olddata = json.load(json_file)
    for ob in olddata['intents']:
        if not (ob["tag"] in detect_duplicate_by_tag):
            intents['intents'].append(ob)
            detect_duplicate_by_tag.append(ob["tag"])

print("old data:", len(intents['intents']))
results = db.collection('users').document(
    'Peo5kqpi4GORXehD3oQVRXpHGfD2').collection('chats').get()
for index, result in enumerate(results):
    data = result.to_dict()
    if not (f"firebaseData{index}" in detect_duplicate_by_tag):
        intents["intents"].append({
            "tag": f"firebaseData{index}",
            "patterns":   [data["messageQuestion2"],data["messageQuestion3"],data["messageQuestion4"],data["messageQuestion5"],data["messageQuestion6"],data["messageQuestion7"]],
            "responses": [data["IAmessageQuestion8"]]
        })
        detect_duplicate_by_tag.append(f"firebase data{index}")

print("new data: ", len(intents['intents']))
with open("intents.json", "w") as outfile:
    json.dump(intents, outfile)
