import os
from dotenv import load_dotenv

load_dotenv()


RUNNINGHUB_API_KEY = os.getenv("RUNNINGHUB_API_KEY")
print(RUNNINGHUB_API_KEY)