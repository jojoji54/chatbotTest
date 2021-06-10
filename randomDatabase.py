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

with open('intents.json') as json_file:
    olddata = json.load(json_file)
    for ob in olddata['intents']:
        intents['intents'].append(ob)

print("old data:", len(intents['intents']))
results = db.collection('users').document(
    'Peo5kqpi4GORXehD3oQVRXpHGfD2').collection('chatsRandom').get()
for index, result in enumerate(results):
    data = result.to_dict()
    intents["intents"].append({
        "tag": f"firebaseRandomData{index}",
        "patterns":   [data["message"]],
        "responses":  [data["IAmessageAnswer1"],data["IAmessageAnswer2"],data["IAmessageAnswer3"],data["IAmessageAnswer4"],data["IAmessageAnswer5"],data["IAmessageAnswer6"]],
    })
print("new data: ", len(intents['intents']))
with open("intents.json", "w") as outfile:
    json.dump(intents, outfile)
