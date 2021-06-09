import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import json
import os

cred = credentials.Certificate("charlotapp-firebase-adminsdk-y8n29-630f0f23a9.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

#read data
#Getting a Document with a known ID
    
intents = {"intents": []}
results = db.collection('users').document('Peo5kqpi4GORXehD3oQVRXpHGfD2').collection('chats').get()

#---------------------------------------
# don' t need this 
# with open('intents.json') as json_file:
#    data = json.load(json_file)
#----------------------------------------


for index, result in enumerate(results):
    data = result.to_dict()
    intents["intents"].append({
        "tag": f"firebase data{index}",
        "patterns":   [data["messageQuestion2"],data["messageQuestion3"],data["messageQuestion4"],data["messageQuestion5"],data["messageQuestion6"],data["messageQuestion7"]],
        "responses": [data["IAmessageQuestion8"]]
    })


print("Log intents: ",intents)

#-------with the new part


#First be sure the file exists and it's not empty. At least it must be structured like this: {"intents":[]}

with open('intents.json') as json_file:
    data = json.load(json_file)

print("Log data: ",data)

intents = {"intents": data["intents"]+intents["intents"]} 


print("Log new intents: ",intents)


with open("intents.json", "w") as outfile: 
    json.dump(intents, outfile)
