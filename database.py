from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017/')
db=client['chatbot_db']
messages_collection=db['messages']
memories_collection=db['memories']
client.admin.command('ping')
print("Connected to MongoDB successfully!")