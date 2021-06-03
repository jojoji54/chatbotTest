import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import json

cred = credentials.Certificate("charlotapp-firebase-adminsdk-y8n29-630f0f23a9.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

#read data
#Getting a Document with a known ID


    
intents = {"intents": []}
results = db.collection('users').document('HtregtuuDDVWglz9DjobFGH9jMo1').collection('chats').get()
for index, result in enumerate(results):
    data = result.to_dict()
    intents["intents"].append({
        "tag": f"firebase data{index}",
        "patterns": [data["message"]],
        "responses": [data["message"]]
    })
#print(intents)
with open("UserMessages.json", "w") as outfile: 
    json.dump(intents, outfile)