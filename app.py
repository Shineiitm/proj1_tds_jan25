import json
import os
import sqlite3

import requests
import pandas as pd
from typing import Dict, Any, List  # Import necessary types
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import mimetypes
from pathlib import Path
from dateutil import parser
import numpy as np
import os
from fastapi import FastAPI
from typing import Dict, Any
import base64
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import subprocess
from fastapi import FastAPI, HTTPException
from dateutil.parser import parse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import requests
import subprocess
import pandas as pd
import markdown
import sqlite3
from bs4 import BeautifulSoup
from PIL import Image
import ffmpeg

load_dotenv()

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Ensure the token is available
if not AIPROXY_TOKEN:
    raise ValueError("❌ AIPROXY_TOKEN is not set. Please add it to your .env file.")
else:
    print(f"✅ AIPROXY_TOKEN Loaded: {AIPROXY_TOKEN[:5]}****")


### 🔹 TASK A1: Generate Data Using `datagen.py`
### 🔹 TASK A1: Generate Data Using `datagen.py`
@app.post("/a1")
def a1(email: str, output_dir: str):
    try:
        subprocess.run(["uv", "run", "datagen.py", email, "--root", output_dir], capture_output=True, text=True, check=True)
        return {"message": "Data generated successfully.", "output_dir": output_dir}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error: {e.stderr}")


### 🔹 TASK A2: Format Markdown File Using Prettier
@app.post("/a2")
def a2(input_file: str = "./data/format.md", output_file: str = "./data/format-formatted.md"):
    file_path = Path(input_file)
    output_path = Path(output_file)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"❌ File {input_file} not found.")

    try:
        # Run Prettier on the input file
        subprocess.run(
            ["npx", "prettier", "--write", str(file_path)],
            capture_output=True, text=True, check=True
        )

        # Read the formatted content and save it
        with open(file_path, "r") as f:
            formatted_content = f.read()

        with open(output_path, "w") as f:
            f.write(formatted_content)

        return {
            "message": f"✅ Formatted {input_file} and saved to {output_file}",
            "output_file": str(output_path),
            "content": formatted_content
        }

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"❌ Prettier Error: {e.stderr}")


### 🔹 TASK A3: Count Wednesdays in Dates File
@app.post("/a3")
def a3(input_file: str, output_file: str):
    try:
        with open(input_file, "r") as f:
            dates = f.readlines()

        wednesday_count = sum(1 for date in dates if parse(date.strip()).weekday() == 2)

        with open(output_file, "w") as f:
            f.write(str(wednesday_count))

        return {"message": f"Counted {wednesday_count} Wednesdays.", "result": wednesday_count}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


### 🔹 TASK A4: Sort Contacts JSON File
@app.post("/a4")
def a4(input_file: str, output_file: str):
    try:
        with open(input_file, "r") as f:
            contacts = json.load(f)

        contacts.sort(key=lambda c: (c["last_name"], c["first_name"]))

        with open(output_file, "w") as f:
            json.dump(contacts, f, indent=2)

        return {"message": f"Sorted contacts and saved to {output_file}."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


### 🔹 TASK A5: Extract First Line of Most Recent Log Files
@app.post("/a5")
def a5(log_dir: str, output_file: str):
    try:
        log_files = sorted(Path(log_dir).glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)[:10]
        first_lines = [f.open().readline().strip() for f in log_files]

        with open(output_file, "w") as f:
            f.write("\n".join(first_lines))

        return {"message": f"Extracted first lines of 10 most recent logs."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


### 🔹 TASK A6: Generate Markdown Index JSON
@app.post("/a6")
def a6(doc_dir: str, output_file: str):
    index = {}
    try:
        for md_file in Path(doc_dir).glob("**/*.md"):
            with open(md_file, "r") as f:
                for line in f:
                    if line.startswith("# "):
                        index[str(md_file.relative_to(doc_dir))] = line[2:].strip()
                        break

        with open(output_file, "w") as f:
            json.dump(index, f, indent=2)

        return {"message": f"Markdown index created.", "index": index}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


### 🔹 TASK A7: Extract Email Sender Using LLM
@app.post("/a7")
def a7(input_file: str = "./data/email.txt", output_file: str = "./data/email-sender.txt"):
    if not Path(input_file).exists():
        raise HTTPException(status_code=404, detail=f"❌ File {input_file} not found.")

    try:
        with open(input_file, "r") as f:
            email_content = f.read()

        print(f"📩 Debug: Email Content Sample: {email_content[:200]}")  # Show sample of email

        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {AIPROXY_TOKEN}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "Extract the sender's email address from the email text provided."},
                    {"role": "user", "content": email_content},
                ]
            }
        )

        print(f"📡 API Response Status: {response.status_code}")  # Debugging API response
        print(f"📡 API Response Text: {response.text}")

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"⚠️ API failed: {response.text}")

        # Extracting sender's email
        response_json = response.json()
        sender_email = response_json["choices"][0]["message"]["content"].strip()

        if not sender_email:
            raise HTTPException(status_code=500, detail="⚠️ No email extracted.")

        with open(output_file, "w") as f:
            f.write(sender_email)

        print(f"✅ Extracted email: {sender_email}")

        return {"message": f"✅ Extracted sender email: {sender_email}"}

    except Exception as e:
        print(f"❌ Error: {str(e)}")  # Debugging any errors
        raise HTTPException(status_code=500, detail=f"❌ Error: {str(e)}")


### 🔹 TASK A8: Extract Credit Card Number from Image
@app.post("/a8")
def a8(image_path: str = "./data/credit_card.png", output_file: str = "./data/credit-card.txt"):
    if not Path(image_path).exists():
        raise HTTPException(status_code=404, detail=f"❌ File {image_path} not found.")

    try:
        with open(image_path, "rb") as image_file:
            image_b64 = base64.b64encode(image_file.read()).decode()

        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/completions",
            headers={"Authorization": f"Bearer {AIPROXY_TOKEN}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "Extract the credit card number from the image provided."},
                    {"role": "user", "content": f"Image data (Base64 encoded): {image_b64}"},
                ]
            }
        )

        print(f"📡 Response Status: {response.status_code}")
        print(f"📡 Response Text: {response.text}")

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"⚠️ API failed: {response.text}")

        card_number = response.json()["choices"][0].get("text", "").strip().replace(" ", "")

        if not card_number:
            raise HTTPException(status_code=500, detail="⚠️ No credit card number extracted.")

        with open(output_file, "w") as f:
            f.write(card_number)

        return {"message": f"✅ Extracted credit card number: {card_number}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error: {str(e)}")



### 🔹 TASK A9: Find Most Similar Comments
@app.post("/a9")
def a9(input_file: str = "./data/comments.txt", output_file: str = "./data/comments-similar.txt"):
    if not Path(input_file).exists():
        raise HTTPException(status_code=404, detail=f"❌ File {input_file} not found.")

    try:
        with open(input_file, "r") as f:
            comments = [line.strip() for line in f.readlines()]

        print(f"📝 First comments: {comments[:3]}")

        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/embeddings",
            headers={"Authorization": f"Bearer {AIPROXY_TOKEN}"},
            json={"model": "text-embedding-3-small", "input": comments}
        )

        print(f"📡 Response Status: {response.status_code}")
        print(f"📡 Response Text: {response.text}")

        if response.status_code == 401:
            raise HTTPException(status_code=401, detail="❌ Unauthorized: Check AIPROXY_TOKEN.")

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"⚠️ API failed: {response.text}")

        response_json = response.json()
        if "data" not in response_json:
            raise HTTPException(status_code=500, detail=f"⚠️ Invalid response: {response_json}")

        embeddings = np.array([entry["embedding"] for entry in response_json["data"]])
        similarity_matrix = np.dot(embeddings, embeddings.T)

        np.fill_diagonal(similarity_matrix, -np.inf)
        i, j = np.unravel_index(similarity_matrix.argmax(), similarity_matrix.shape)

        similar_comments = f"{comments[i]}\n{comments[j]}"

        with open(output_file, "w") as f:
            f.write(similar_comments)

        return {"message": "✅ Most similar comments found.", "comments": similar_comments}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error: {str(e)}")



### 🔹 TASK A10: Calculate Total Sales for "Gold" Tickets
@app.post("/a10")
def a10(db_path: str, output_file: str):
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM tickets WHERE type='Gold'", conn)
        total_sales = (df["units"] * df["price"]).sum()

        with open(output_file, "w") as f:
            f.write(str(total_sales))

        return {"message": f"Total Gold ticket sales: {total_sales}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


tools: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "a1",
            "description": "Generate required data using datagen.py",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "User's email address used for generating specific data"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Path to the output directory where generated data files will be stored"
                    }
                },
                "required": ["email", "output_dir"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "a2",
            "description": "Format a Markdown file using Prettier",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "Path to the Markdown file that needs formatting"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to save the formatted Markdown file"
                    }
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "a3",
            "description": "Count the number of Wednesdays from a list of dates",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "Path to the file containing a list of dates, one per line"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to save the count of Wednesdays found in the input file"
                    }
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "a4",
            "description": "Sort contacts JSON file by last name and first name",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "Path to the contacts JSON file containing an array of contacts"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to save the sorted contacts JSON file"
                    }
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "a5",
            "description": "Extract the first line from the 10 most recent log files",
            "parameters": {
                "type": "object",
                "properties": {
                    "log_dir": {
                        "type": "string",
                        "description": "Directory containing log files, which should be sorted by modification time"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to save the extracted first lines from the 10 most recent log files"
                    }
                },
                "required": ["log_dir", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "a6",
            "description": "Generate an index of Markdown files by extracting their first H1 heading",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_dir": {
                        "type": "string",
                        "description": "Directory containing Markdown (.md) files"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to save the JSON index mapping file names to their H1 headings"
                    }
                },
                "required": ["doc_dir", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "a7",
            "description": "Extract the sender's email address from an email file",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "Path to the email text file that contains email content"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to save the extracted sender email address"
                    }
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "a8",
            "description": "Extract a credit card number from an image using OCR or an AI model",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the credit card image file"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to save the extracted credit card number as plain text"
                    }
                },
                "required": ["image_path", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "a9",
            "description": "Find the most similar pair of comments using embeddings",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "Path to the file containing a list of comments, one per line"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to save the most similar comments based on cosine similarity"
                    }
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "a10",
            "description": "Calculate the total sales of 'Gold' tickets from a SQLite database",
            "parameters": {
                "type": "object",
                "properties": {
                    "db_path": {
                        "type": "string",
                        "description": "Path to the SQLite database file containing ticket sales data"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to save the total sum of sales for 'Gold' ticket type"
                    }
                },
                "required": ["db_path", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a specified file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    }
                },
                "required": ["path"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "fetch_data_from_api",
            "description": "Fetch data from an API and save it to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the API endpoint to fetch data from"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to save the fetched API response"
                    }
                },
                "required": ["url", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "clone_git_repo",
            "description": "Clone a Git repository to a specified destination",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_url": {
                        "type": "string",
                        "description": "URL of the Git repository to clone"
                    },
                    "destination": {
                        "type": "string",
                        "description": "Path where the repository should be cloned"
                    }
                },
                "required": ["repo_url", "destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql_query",
            "description": "Run an SQL query on a database file",
            "parameters": {
                "type": "object",
                "properties": {
                    "database_file": {
                        "type": "string",
                        "description": "Path to the SQLite database file"
                    },
                    "query": {
                        "type": "string",
                        "description": "SQL query to be executed"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to save the query results (optional)"
                    }
                },
                "required": ["database_file", "query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_website",
            "description": "Extract text content from a website",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the website to scrape"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to save the extracted website text"
                    }
                },
                "required": ["url", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "resize_image",
            "description": "Resize an image to specified dimensions",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_image": {
                        "type": "string",
                        "description": "Path to the input image file"
                    },
                    "output_image": {
                        "type": "string",
                        "description": "Path to save the resized image"
                    },
                    "width": {
                        "type": "integer",
                        "description": "New width for the image (optional)"
                    },
                    "height": {
                        "type": "integer",
                        "description": "New height for the image (optional)"
                    }
                },
                "required": ["input_image", "output_image"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "transcribe_audio",
            "description": "Convert audio file to text",
            "parameters": {
                "type": "object",
                "properties": {
                    "audio_file": {
                        "type": "string",
                        "description": "Path to the input audio file"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to save the transcribed text"
                    }
                },
                "required": ["audio_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_markdown_to_html",
            "description": "Convert a Markdown file to an HTML file",
            "parameters": {
                "type": "object",
                "properties": {
                    "markdown_file": {
                        "type": "string",
                        "description": "Path to the input Markdown file"
                    },
                    "output_html": {
                        "type": "string",
                        "description": "Path to save the converted HTML file"
                    }
                },
                "required": ["markdown_file", "output_html"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filter_csv",
            "description": "Filter a CSV file based on column value and save the results as JSON",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_file": {
                        "type": "string",
                        "description": "Path to the CSV file"
                    },
                    "filter_column": {
                        "type": "string",
                        "description": "Column name to apply the filter on"
                    },
                    "filter_value": {
                        "type": "string",
                        "description": "Value to filter the column by"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to save the filtered data as JSON"
                    }
                },
                "required": ["csv_file", "filter_column", "filter_value", "output_file"]
            }
        }
    }


]



 # Initialize an empty list if not provided


def query_gpt(user_input: str):

    response = requests.post(
        "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Whenever you receive a system directory location, if there is no / or . in the start add a ./ at the start of the path, if only / is there in start add a . in the fronts"},
                {"role": "user", "content": user_input},
            ],
            "tools": tools,
            "tool_choice": "auto",
        },
    )
    return response.json()

@app.post("/run")
async def run(task: str):
    query = query_gpt(task)
    print(query)
    try:
        func_name = query["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
        args = json.loads(query["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        func = eval(func_name)  # ⚠️ Be careful with eval()
        output = func(**args)
        return output
    except Exception as e:
        return {"error": str(e)}


@app.get("/read")
async def read(path: str):
    # with open(path,"r") as f:
    #     data = f.read()
    # return data
    # Ensure the path is formatted correctly
    if not path.startswith("/") and not path.startswith("./"):
        path = "./" + path
    elif path.startswith("/") and not path.startswith("./"):
        path = "." + path

    # Convert path to absolute form
    file_path = Path(path).resolve()

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return StreamingResponse(file_path.open("r", encoding="utf-8"), media_type=mimetypes.guess_type(path)[0])








