import os
import json
import pandas as pd
from fastapi import APIRouter

router = APIRouter()

# Define the paths to your directories
# This sets the base directory to the project root
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
DATA_DIR = os.path.join(BASE_DIR, "data")
JSON_DIR = os.path.join(BASE_DIR, "json_data")

# Create the output directory if it doesn't exist
os.makedirs(JSON_DIR, exist_ok=True)

@router.get("/convert-and-save-all-data")
async def convert_and_save_all_data():
    """
    Converts all CSV files to JSON and saves them to the json_data directory.
    """
    files_to_convert = {
        "posts_reddit.json": "posts Data Dump - Reddit.csv",
        "posts_youtube.json": "posts Data Dump - Youtube.csv",
        "comments_reddit.json": "comments Data Dump - Reddit.csv",
        "comments_youtube.json": "comments Data Dump - Youtube.csv",
    }
    
    conversion_results = {}
    for json_filename, csv_filename in files_to_convert.items():
        csv_path = os.path.join(DATA_DIR, csv_filename)
        json_path = os.path.join(JSON_DIR, json_filename)
        
        try:
            df = pd.read_csv(csv_path)
            # Convert DataFrame to a list of dictionaries
            json_data = df.to_dict(orient="records")
            
            # Save the JSON data to a file
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=4)
            
            conversion_results[csv_filename] = "Success"
        except FileNotFoundError:
            conversion_results[csv_filename] = "Failed (File not found)"
        except Exception as e:
            conversion_results[csv_filename] = f"Failed (Error: {str(e)})"
            
    return {"message": "Conversion process completed.", "results": conversion_results}

# Keep the original endpoints for serving data
@router.get("/posts/reddit")
async def get_reddit_posts():
    file_path = os.path.join(DATA_DIR, "posts Data Dump - Reddit.csv")
    try:
        df = pd.read_csv(file_path)
        return df.to_dict(orient="records")
    except FileNotFoundError:
        return {"error": "File not found."}, 404
# ... (you would add the other three original endpoints here as well)
# I'll just keep this one here for simplicity.
