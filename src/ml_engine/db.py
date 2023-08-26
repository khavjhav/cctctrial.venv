from pymongo import MongoClient
import os
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()


DB_HOST = os.getenv("DB_HOST")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
uri = (
    f"mongodb+srv://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}/?retryWrites=true&w=majority"
)
client = MongoClient(uri, server_api=ServerApi("1"))