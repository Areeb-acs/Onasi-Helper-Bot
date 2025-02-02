from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from backend.core import run_llm
from langchain_groq import ChatGroq
import json
import os
from dotenv import load_dotenv
import requests
import base64
import logging
from uuid import uuid4  # For generating unique session IDs
# from langchain_pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from fastapi.responses import JSONResponse
from fastapi import HTTPException
import requests
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from fastapi.responses import HTMLResponse, PlainTextResponse
from backend.sql_queries import fetch_query_results

import re


# Load environment variables
load_dotenv()

# GitHub Configuration
# GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
# REPO_OWNER = os.getenv("REPO_OWNER")
# REPO_NAME = os.getenv("REPO_NAME")
# FILE_PATH = os.getenv("FILE_PATH")
# BRANCH = os.getenv("BRANCH", "main")

app = FastAPI()
# File to store conversations
# CONVERSATION_LOG_FILE = "conversations.txt"
SUPPORTED_DOMAINS = {"RCM", "DHIS"}

# A global dictionary to store session-specific chat histories in memory
session_chat_histories = {}


# BUCKET_NAME = "onasi-chatbot"
# FILE_URL = "conversation.txt"
INDEX_NAME = ""
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

groq_api_key = os.getenv("GROQ_API_KEY")
# Initialize Pinecone

# Load the FAISS index
faiss_index_path = "faiss_index"  # Path to your FAISS index directory
docsearch = FAISS.load_local(
    folder_path=faiss_index_path,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)# Create a retriever from the FAISS vector store
# docsearch = Pinecone(index_name=INDEX_NAME, embedding=embeddings)


def update_local_file(new_content, file_path="conversation.txt"):
    """
    Append new content to a local file.
    
    Args:
    new_content (str): The new content to append to the file.
    file_path (str): Path to the local file. Defaults to 'conversation.txt'.
    """
    try:
        # Read current content from the file if it exists
        try:
            with open(file_path, "r") as file:
                current_content = file.read()
        except FileNotFoundError:
            # If the file doesn't exist, start with empty content
            current_content = ""

        # Append new content
        updated_content = current_content + new_content

        # Write the updated content back to the file
        with open(file_path, "w") as file:
            file.write(updated_content)

        logging.info(f"File '{file_path}' updated successfully!")
    except Exception as e:
        logging.error(f"Error updating local file '{file_path}': {str(e)}")

import time


# def get_last_10_conversations():
#     """
#     Fetches the last 10 Q&A pairs from the S3 conversations file.

#     Returns:
#         List[dict]: A list of the last 10 conversations in the format:
#                     [{"user": "question1", "ai": "answer1"}, ...]
#     """
#     try:
#         # Fetch the content of the file from the S3 bucket
#         response = requests.get(FILE_URL)
#         if response.status_code == 200:
#             content = response.text
#         else:
#             logging.error(f"Failed to fetch conversation file: {response.status_code}")
#             return []

#         # Split content into individual conversations by the separator
#         entries = content.strip().split("==================================================")
#         conversations = []

#         for entry in entries:
#             # Process each entry to extract User and AI lines
#             lines = entry.strip().split("\n")
#             user_line = next((line.replace("User: ", "").strip() for line in lines if line.startswith("User:")), None)
#             ai_line = next((line.replace("AI: ", "").strip() for line in lines if line.startswith("AI:")), None)

#             # Append only valid entries with both User and AI content
#             if user_line and ai_line:
#                 conversations.append({"user": user_line, "ai": ai_line})

#         # Return the last 10 conversations
#         return conversations[-1:] if len(conversations) > 1 else conversations

#     except Exception as e:
#         logging.error(f"Error fetching or parsing conversation file: {str(e)}")
#         return []

start_time = time.time()
print(f"QA chain execution took: {time.time() - start_time:.2f} seconds")

# Initialize ChatGroq
groq_api_key = os.getenv("GROQ_API_KEY")

chat = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192") # deepseek-r1-distill-llama-70b

# Preinitialize docsearch (can be reused across multiple queries)
# docsearch = Pinecone(index_name=INDEX_NAME, embedding=embeddings)

# HTML formatting prompt template
HTML_PROMPT_TEMPLATE = """
Please use the answer and keep it exactly the same, just change to html format ONLY.
Answer to format: {answer}
"""


# Load FAQ data
with open("./faq_data.json", "r") as f:
    faq_data = json.load(f)






def get_conversation_by_session_id(session_id, recent_count=4, file_path="conversation.txt"):
    """
    Fetches the last few user queries and AI responses for a specific session ID from a local file.

    Args:
        session_id (str): The session ID to filter conversations by.
        recent_count (int): Number of most recent conversations to return.
        file_path (str): Path to the local conversation file.

    Returns:
        List[dict]: A list of the last `recent_count` conversations in the format:
                    [{"user": "question1", "ai": "response1"}, ...]
    """
    try:
        # Read the content of the file
        try:
            with open(file_path, "r") as file:
                content = file.read()
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return []

        # Check if the session ID exists
        session_marker = f"New Session Initialized: {session_id}"
        if session_marker not in content:
            logging.info(f"Session ID {session_id} not found.")
            update_local_file(f"New Session Initialized: {session_id}\n{'=' * 50}\n")
            return []

        # Split content into lines and process only the relevant session
        lines = content.splitlines()
        conversations = []
        in_session = False
        current_user = None
        current_ai = []

        for line in lines:
            line = line.strip()

            if line.startswith("New Session Initialized:"):
                # Check if this is the target session
                if line == session_marker:
                    in_session = True  # Start capturing this session
                    logging.info(f"Started capturing session: {session_id}")
                elif in_session:
                    # Stop capturing if a new session starts
                    logging.info("Reached the next session marker. Stopping capture.")
                    break

            elif in_session:
                if line.startswith("User:"):
                    # Save the previous conversation if a new user query starts
                    if current_user and current_ai:
                        conversations.append({"user": current_user, "ai": "\n".join(current_ai)})
                        current_ai = []

                    # Start a new user query
                    current_user = line.replace("User: ", "").strip()

                elif line.startswith("AI:"):
                    # Collect all AI response lines
                    current_ai.append(line.replace("AI: ", "").strip())

                else:
                    # Append multi-line AI responses
                    if current_ai:
                        current_ai.append(line)

        # Append the final conversation if it exists
        if current_user and current_ai:
            conversations.append({"user": current_user, "ai": "\n".join(current_ai)})

        # Keep only the most recent conversations
        logging.info(f"Captured conversations: {conversations}")
        return conversations[-recent_count:]

    except Exception as e:
        logging.error(f"Error fetching or parsing conversation file: {str(e)}")
        return []
def log_conversation(user_query, ai_response):
    """Logs the conversation to the public S3 file."""
    new_entry = f"\nUser: {user_query}\nAI: {ai_response}\n"
    update_local_file(new_entry)
    
def format_chat_history(chat_history):
    """
    Format the chat history into a readable string for inclusion in LLM input.
    """
    if not chat_history:
        return "No previous history."

    formatted_history = ""
    for entry in chat_history:
        user_input = entry.get("user", "Unknown input")
        ai_response = entry.get("ai", "Unknown response")
        formatted_history += f"User: {user_input}\nAI: {ai_response}\n"
        # formatted_history += f"User: {user_input}\n"
    
    return formatted_history.strip()

@app.get("/")
async def get_root():
    return {"message": "Welcome to Onasi Helper Bot!"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    """
    Chat endpoint for processing user queries and returning AI-generated responses.

    Workflow:
    1. Parse incoming request data.
    2. Check for an existing session or initialize a new one.
    3. Match the query against FAQ data for faster responses.
    4. If not found in FAQ, proceed with LLM response generation.
    5. Log the conversation to S3 (batched for efficiency).
    6. Stream the response back to the user.
    """

   

    data = await request.json()
    # print(data)
    question = data.get("question")
    session_id = data.get("session_id")
    domain = data.get("domain", None)
    chat_history = get_conversation_by_session_id(session_id)
    print(format_chat_history(chat_history))

    if not question:
        return {"error": "Question is required."}

    # ------------------------------
    # 1. Initialize or Fetch Session
    # ------------------------------
    if not chat_history:
        logging.info(f"Generated new session ID: {session_id}")

        # Log new session initialization
        

    # ------------------------------
    # 2. Determine Domain
    # ------------------------------
    if not domain:
        if "rcm" in question.lower():
            domain = "RCM"
        elif "dhis" in question.lower():
            domain = "DHIS"
        else:
            domain = None

    if domain and domain.upper() not in SUPPORTED_DOMAINS:
        return {"error": f"Unsupported domain '{domain}'."}

    # ------------------------------
    # 3. Optimize FAQ Matching
    # ------------------------------
    # Use a dictionary for O(1) lookup instead of iterating through the list
    faq_lookup = {qa["question"].lower(): qa["answer"] for qa in faq_data}
    raw_answer = faq_lookup.get(question.lower())
    if raw_answer:

        # Format FAQ response into HTML
        formatted_response = chat.invoke(HTML_PROMPT_TEMPLATE.format(answer=raw_answer))
        return {
            "response": HTMLResponse(content=formatted_response.content)
        }

    # ------------------------------
    # 4. LLM Response Generation
    # ------------------------------
    async def response_generator():
        """
        Generate the response using LLM with streaming.
        """
        # conversation_data = get_last_10_conversations()  # Fetch last 10 Q&A pairs from S3
        import time

        start_time = time.time()
        try:
            # Pass chat history and other params to `run_llm`
            generated_response = run_llm(
                query=question,
                chat=chat,
                docsearch=docsearch,
                chat_history=format_chat_history(chat_history),
                domain=domain
            )
                
            print(f"QA chain execution took: {time.time() - start_time:.2f} seconds")


            # Handle different types of `generated_response`
            if isinstance(generated_response, str):
                # If the response is a string, treat it as the full answer
                answer = generated_response

            elif isinstance(generated_response, dict):
                # If the response is a dictionary, extract the "answer" key
                answer = generated_response.get("answer", "")

            elif hasattr(generated_response, "content"):
                # If the response is an object with a "content" attribute
                answer = generated_response.content

            else:
                # Unexpected response type
                raise TypeError(f"Unexpected response type: {type(generated_response)}")
                
            # Remove content between <think> tags
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)

            # Log the conversation for debugging/auditing
            log_conversation(question, answer)

            # Stream response chunks
            for chunk in answer:
                yield chunk

        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            yield "An error occurred while generating the response."


    logging.info(f"Processing query: {question}, Domain: {domain}")
    return StreamingResponse(response_generator(), media_type="text/plain")







# Print the extracted ClaimSection data
# Define the function
# Define the function
# Define the function
def check_claim_section(claim_section_data):
    """Check for required keys in ClaimSection using ChatGroq."""
    # Convert claim_section_data to JSON string
    fhir_message = json.dumps(claim_section_data, indent=4)

    # Define the prompt
    prompt_template = PromptTemplate(
        template="""You will always output as HTML. Please start of answer of with: Analyzing claim submission: 
FHIR Message:
{fhir_message}

Tasks:

1. You are an expert in analyzing the text and output in HTML format. Format it neatly.Please take the text as it is. Do not hallucinate.
2. Avoid using <html> tags, avoid markdown, always output in bullet points using <li> and <ul> tags.
3. ALways be direct, and output neatly the results.
4. Just take the answer, and output only in HTML format. Do not add anything from yourself.
5. If there are errors found in the claim submission output, please explain them to the user simply.
6. Please do not say Clincal Claim
7. Please use the output above, do not add anything like extra thigns. Just just convert them to HTML format.
8. If an error is shown, please explain simply to the user what needs to be done, this is related to the RCM application for medical insurance. 
9. In error explanation and solution, please do not use thr word sequence, do not use any technical jargon, just use simple wording, what the user has to do.
10. When formatting in bullet points, I think it is best if there is no Error shown, you show tick marks, and if Error shown, then so cross mark.
11. Tick marks should be in green and the response should be in green, the Error response and the cross mark should be in red.
12. Please ensure the the JSON output is correct, JSON format and there is no unescaped quotation marks.
13. Avoid using Markdown for line breaks, just use html syntax only. No line breaks.
14. Please make sure not to repeat yourself. summarize answer neatly for end user.


As an example, please use the below sample output to take guidance from.
If all checks have passed, you will output:

Analyzing claim submission:
    
<ul>
    <li><b>ClaimSection:</b> <font color=green>✓ All Checks Passed</font></li>
    <li><b>CareTeamSection:</b> <font color=green>✓ All Checks Passed</font></li>
    <li><b>SupportingInformationSection:</b> <font color=green>✓ All Checks Passed</font></li>
    <li><b>DiagnosisSection:</b> <font color=green>✓ All Checks Passed</font></li>
    <li><b>ProductService:</b> <font color=green>✓ All Checks Passed</font></li>
    <li><b>ProductService: ICD-10 and CPT Check</b> <font color=green>✓ All Checks Passed</font></li>
    <li><b>EncounterSection:</b> <font color=green>✓ All Checks Passed</font></li>
</ul>
 
 
 RULES: 
 1. NEVER USE CARETEAM for supportinginformation section.
 If there is an error somewhere, you can do like the following, please note, the following is an example only:
 For errors, like for example, I forgot to add the care team sequence information in Product & Services section:
 
 Analyzing claim submission:
<ul>
  <li><b>ClaimSection:</b> <font color="green">✓ All Checks Passed</font></li>
  <li><b>CareTeamSection:</b> <font color="green">✓ All Checks Passed</font></li>
  <li><b>SupportingInformationSection:</b> <font color="green">✓ All Checks Passed</font></li>
  <li><b>DiagnosisSection:</b> <font color="green">✓ All Checks Passed</font></li>
  <li><b>ProductService:</b> <font color="red">✗ Error: Missing key(s) careTeam</font>
    <ul>
      <li>Explanation: This error means that there is a missing key for the careTeam in the ProductService, which is required for the claim submission.</li>
      <li>Solution: Please ensure that you have provided the required careTeam key in the ProductService.</li>
    </ul>
  </li>
  <li><b>ProductService: ICD-10 and CPT Check</b> <font color=green>✓ All Checks Passed</font></li>

</ul>

    
    

""",
        input_variables=["fhir_message"]
    )

    # Generate the prompt
    prompt = prompt_template.format(fhir_message=fhir_message)
    
    # Invoke the model
    print("Sending request to ChatGroq...")
    response = chat.invoke(prompt)
    print("Response received from ChatGroq.")
    
    # Debugging: Print the raw response
    print("Raw response from ChatGroq:", response)
    print("Type of response:", type(response))
    
    # Ensure response has content
    if not hasattr(response, 'content') or not isinstance(response.content, str):
        return "<font color='red'>✗ Error: Invalid or missing response from ChatGroq.</font>"
    
    # Extract the content attribute
    raw_content = response.content
    
    # Remove <think> tags and everything in between them
    cleaned_response = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
    
    return cleaned_response



from fastapi.responses import Response


@app.post("/analyze-fhir")
async def analyze_fhir_endpoint(request: Request):
    """
    API endpoint to analyze FHIR messages and explain the results.

    Expects the FHIR JSON file as input in the request body.

    Returns:
    {
        "sections": {<Extracted sections>},
        "validation_results": {<Section validations>}
    }
    """
    try:
        # Load JSON data from request body
        fhir_content = await request.json()
        if not fhir_content:
            raise HTTPException(status_code=400, detail="FHIR JSON content is required in the request body.")
        
        logging.info("FHIR JSON content loaded successfully.")

        # Extract sections
        sections = extract_sections(fhir_content)
        if not sections:
            raise HTTPException(status_code=400, detail="No sections extracted from FHIR content.")

        # Sequentially validate sections and stop if an error is found
        validation_results = {}
        sections_to_validate = [
            ("ClaimSection", ["status", "type", "subType", "use", "patient", "created", "insurer", "provider", "priority", "payee"]),
            ("CareTeamSection", ["sequence", "provider", "role", "qualification"]),
            ("SupportingInformationSection", []),
            ("DiagnosisSection", ["sequence", "diagnosisCodeableConcept", "type"]),
            ("ProductService", [
                "extension", "sequence", "careTeamSequence", "diagnosisSequence",
                "informationSequence", "productOrService", "quantity", "unitPrice", "net"
            ]),
            ("EncounterSection", [
                "resourceType", "identifier", "status", "serviceType", "priority", "subject", "period", "serviceProvider"
            ])
        ]
        
        for section_name, required_keys in sections_to_validate:
            validation_results[section_name] = validate_section(sections.get(section_name, {}), required_keys, section_name)

            # Stop further validation if any error or missing keys are found
            if "Error" in validation_results[section_name] or "Missing" in validation_results[section_name]:
                break



        # Fetch the procedural code explanation for the given sections
        # This function retrieves the description of the procedural code (CPT) based on its system and code values.
        # The procedural code explanation will help determine whether it aligns with the diagnosis.
        procedural_code_explanation = get_procedural_code_explanation(sections)

        # Fetch the diagnosis code value and its display value
        # This function retrieves the ICD-10 code value and its corresponding display value from the database.
        # The diagnosis result will be used to validate the alignment with the procedural code.
        diagnosis_result = fetch_code_value_and_display_value(sections)

        # Run the coding consistency check
        # This step compares the diagnosis description (ICD-10) and the procedural code description (CPT).
        # The `check_coding_consistency` function leverages AI to identify any mismatches or inconsistencies
        # and generates an HTML-formatted response with results.
        procedure_description = procedural_code_explanation  # Use the procedural code explanation as the procedure description
        consistency_output = check_coding_consistency(
            diagnosis_description=diagnosis_result,  # The diagnosis description (ICD-10) retrieved earlier
            procedure_description=procedure_description,  # The procedural code description (CPT) retrieved earlier
        )

        # Append the coding consistency results to the ProductService validation results
        # The following steps ensure that the validation results for 'ProductService' are stored in a consistent format:
        # 1. If the current value in `validation_results['ProductService']` is a string (e.g., an error message),
        #    convert it into a list to allow appending multiple results.
        # 2. Append the output of the consistency check to the `ProductService` results.

        # Check if ProductService exists and initialize it as an empty list if it doesn't
        if 'ProductService' not in validation_results:
            validation_results['ProductService'] = []
            validation_results['ProductService'].append("Error: ProductService section is empty.")


        else:
            # If the ProductService entry is a string (e.g., an error message), convert it into a list
            if isinstance(validation_results['ProductService'], str):
                validation_results['ProductService'] = [validation_results['ProductService']]
                
            # Append the result of the consistency check (HTML output from AI Helper)
            validation_results['ProductService'].append({
                "message": consistency_output,
                "details": "This is related to ICD-10 and CPT validation check."
            })

            



        if not validation_results:
            raise HTTPException(status_code=400, detail="Validation results are empty.")

        # Process claim section and prepare response
        # Process claim section and prepare response
        final_html = check_claim_section(validation_results)  # This already returns a string
        
        # Remove unwanted characters (quotes, newlines) from the final HTML
        result = re.sub(r"[\"'\n]", "", final_html.strip())
        
        logging.info("Analysis completed successfully.")
        
        # Return JSON object
        return JSONResponse(content=result)

    except json.JSONDecodeError:
        logging.error("Invalid JSON format received.")
        raise HTTPException(status_code=400, detail="Invalid JSON format.")
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")
    

def validate_supporting_info(supporting_info):
    """
    Validate the supportingInfo section with sequence-specific checks.

    Parameters:
    - supporting_info (list): List of supportingInfo entries to validate.

    Returns:
    - str: Validation result, "All Checks Passed" or detailed error messages.
    """
    errors = []

    # Iterate through each supportingInfo entry
    for entry in supporting_info:
        sequence = entry.get("sequence", "Unknown")  # Get the sequence number or default to "Unknown"

        # Define required keys based on the sequence pattern
        if sequence in [1, 10, 15]:
            required_keys = ["sequence", "category", "code"]
        elif sequence in [2, 3, 4, 5, 6, 7, 8, 9]:
            required_keys = ["sequence", "category", "valueQuantity"]
        elif sequence in [11, 12, 13, 14]:
            required_keys = ["sequence", "category", "valueString"]
        else:
            errors.append(f"Validation for sequence {sequence} is not implemented.")
            continue  # Skip further validation for unhandled sequences

        # Validate each required key
        for key in required_keys:
            if key not in entry:
                errors.append(f"Missing key '{key}' in sequence {sequence}")
            elif key == "category":
                # Validate 'category' has 'coding' and required fields
                if not isinstance(entry["category"], dict) or "coding" not in entry["category"]:
                    errors.append(f"Missing or invalid 'coding' in 'category' for sequence {sequence}")
                else:
                    for coding_entry in entry["category"]["coding"]:
                        if "system" not in coding_entry or "code" not in coding_entry:
                            errors.append(f"Missing 'system' or 'code' in 'category.coding' for sequence {sequence}")
            elif key == "code":
                # Validate 'code' has 'coding' and required fields
                if not isinstance(entry["code"], dict) or "coding" not in entry["code"]:
                    errors.append(f"Missing or invalid 'coding' in 'code' for sequence {sequence}")
                else:
                    for coding_entry in entry["code"]["coding"]:
                        if "system" not in coding_entry or "code" not in coding_entry:
                            errors.append(f"Missing 'system' or 'code' in 'code.coding' for sequence {sequence}")
            elif key == "valueQuantity":
                # Validate 'valueQuantity' has 'value', 'system', and 'code'
                if not isinstance(entry["valueQuantity"], dict):
                    errors.append(f"Missing or invalid 'valueQuantity' in sequence {sequence}")
                else:
                    for sub_key in ["value", "system", "code"]:
                        if sub_key not in entry["valueQuantity"]:
                            errors.append(f"Missing '{sub_key}' in 'valueQuantity' for sequence {sequence}")
            elif key == "valueString":
                # Validate 'valueString' exists and is a string
                if not isinstance(entry.get("valueString"), str):
                    errors.append(f"Missing or invalid 'valueString' in sequence {sequence}")

    # Return all error messages or "All Checks Passed" if no errors
    return "\n".join(errors) if errors else "All Checks Passed"

def extract_sections(data):
    """Extract and organize sections from the JSON data."""
    sections = {
        "ClaimSection": [],
        "CareTeamSection": [],
        "SupportingInformationSection": [],
        "DiagnosisSection": [],
        "ProductService": [],
        "EncounterSection": []
    }
    
    keywords = ["meta", "extension", "identifier", "status", "type", "subType", 
                "use", "patient", "created", "insurer", "provider", "priority", "payee"]

    for entry in data.get('entry', []):
        resource = entry.get('resource', {})
        resource_type = resource.get('resourceType')

        if resource_type == 'Claim':
            filtered_claim = {key: resource[key] for key in keywords if key in resource}
            sections["ClaimSection"].append(filtered_claim)

            if 'careTeam' in resource:
                sections["CareTeamSection"].extend(resource['careTeam'])
         # Extract and validate supportingInfo
            if 'supportingInfo' in resource:
                supporting_info = resource['supportingInfo']
                # validation_result = validate_supporting_info(supporting_info)
                # print(f"Supporting Information Validation:\n{validation_result}\n")
                sections["SupportingInformationSection"].extend(supporting_info)

            if 'diagnosis' in resource:
                sections["DiagnosisSection"].extend(resource['diagnosis'])
            if 'item' in resource:
                sections["ProductService"].extend(resource['item'])

        elif resource_type == 'Encounter':
            sections["EncounterSection"].append(resource)

    return sections


def validate_section(section, required_keys, section_name=None):
    """
    Validate a section for required keys or delegate to a specialized validation function if applicable.
    Parameters:
    - section (list): The section to validate.
    - required_keys (list): List of keys expected in each entry of the section.
    - section_name (str): Name of the section (optional, used for specialized handling).
    Returns:
    - str: Validation result, "All Checks Passed" or detailed error messages.
    """
    errors = []

    # Handle missing section
    if not section:
        errors.append(f"Error: Section '{section_name}' not found.")

    # Specialized validation for SupportingInformationSection
    if section_name == "SupportingInformationSection":
        supporting_info_errors = validate_supporting_info(section)
        if supporting_info_errors:
            errors.append(supporting_info_errors)
    else:
        # General validation for other sections
        for index, entry in enumerate(section):
            if isinstance(entry, dict):
                missing_keys = [key for key in required_keys if key not in entry]
                if missing_keys:
                    errors.append(f"Error in entry {index + 1}: Missing key(s) {', '.join(missing_keys)}")
            elif isinstance(section, list) and len(section) > 0 and isinstance(section[0], dict):
                errors.append(f"Error in entry {index + 1}: Invalid section structure.")
            else:
                errors.append(f"Error in entry {index + 1}: Invalid entry type.")

    return "\n".join(errors) if errors else "All Checks Passed"



# AI Helper - Incorrect or Inconsisten Coding - Use case 1:


def fetch_code_value_and_display_value(sections):
    """
    Fetch the CodeValue and CodeDisplayValue for a given CodeValue.
    
    Args:
        code (str): The CodeValue to search for.
    
    Returns:
        str: A formatted string containing both CodeValue and CodeDisplayValue,
             or an error message if not found.
    """

    code = sections['DiagnosisSection'][0]['diagnosisCodeableConcept']['coding'][0]['code']

    # Define the SQL query
    sql_query = f"SELECT CodeValue, CodeDisplayValue FROM Sys_Codes WHERE CodeValue = '{code}'"

    # Fetch the query result
    result = fetch_query_results(sql_query)

    # Parse and return the CodeValue and CodeDisplayValue as a formatted string
    try:
        result_json = json.loads(result)
        if result_json and isinstance(result_json, list) and "CodeValue" in result_json[0] and "CodeDisplayValue" in result_json[0]:
            code_value = result_json[0]["CodeValue"]
            code_display_value = result_json[0]["CodeDisplayValue"]
            return f"CodeValue: {code_value}, CodeDisplayValue: {code_display_value}"
        else:
            return f"No CodeValue or CodeDisplayValue found for CodeValue: {code}"
    except json.JSONDecodeError as e:
        return f"Error parsing result: {e}"
    
    
    
def get_procedural_code_explanation(sections):
    """
    Fetch the explanation of the procedural code from the given sections data.
    
    Args:
        sections (dict): The dictionary containing ProductService and coding information.
    
    Returns:
        str: The explanation for the procedural code or a default message if not found.
    """
    prodecural_codes_mapping = {
        "http://nphies.sa/terminology/CodeSystem/transportation-srca": "This code set includes Ambulance and transportation services (SRCA)",
        "http://nphies.sa/terminology/CodeSystem/imaging": "This code set includes Imaging Procedures",
        "http://nphies.sa/terminology/CodeSystem/laboratory": "This code set includes Laboratory tests, observations and Blood Bank products",
        "http://nphies.sa/terminology/CodeSystem/medical-devices": "This code set includes Medical devices",
        "http://nphies.sa/terminology/CodeSystem/oral-health-ip": "This code set includes Oral Health - In-patient",
        "http://nphies.sa/terminology/CodeSystem/oral-health-op": "This code set includes Oral Health - Out-patient",
        "http://nphies.sa/terminology/CodeSystem/medication-codes": "This code set includes all drug or medicament substance codes and all pharmaceutical products",
        "http://nphies.sa/terminology/CodeSystem/procedures": "This code set includes Procedures / Health interventions",
        "http://nphies.sa/terminology/CodeSystem/services": "This code set includes Room and Board, In-patient Rounding, Consultations, Services",
        "http://nphies.sa/terminology/CodeSystem/body-site": "This code set includes Specific and identified anatomical location of the service provided to the patient (limb, tooth, etc.)"
    }

    try:
        # Extract the link from sections
        link = sections['ProductService'][0]['productOrService']['coding'][0]['system'] 
        print(prodecural_codes_mapping.get(link, "Explanation not found for this link."))
        # Get the explanation from the mapping
        return prodecural_codes_mapping.get(link, "Explanation not found for this link.") + " CPT Code:" +  sections['ProductService'][0]['productOrService']['coding'][0]['code'] 
    except (KeyError, IndexError) as e:
        return f"Error accessing coding data: {e}"
    
    
    
def check_coding_consistency(diagnosis_description, procedure_description):
    """
    Check for consistency between a diagnosis description (ICD-10) and a procedure description (CPT code)
    using ChatGroq as an expert in medical insurance.

    Args:
        diagnosis_description (str): The description of the diagnosis (ICD-10).
        procedure_description (str): The description of the procedure (CPT code).

    Returns:
        str: HTML-formatted response with the analysis of the consistency between the two codes.
    """
    # Define the ChatGroq API Key
    groq_api_key = os.getenv("GROQ_API_KEY")
    chat = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

    # Chat prompt template for analyzing coding consistency
    prompt_template = PromptTemplate(
        template="""
        You are an expert in medical insurance claims validation and medical coding. Your task is to check if the provided diagnosis description (ICD-10) aligns with the procedure description (CPT code).
        Follow these rules:
        1. Just say the result directly, please give as little output as possible.
        2. For some cases like fever or others, we can use both inpatient or outpatient, please do not generate Error in this case.
        3. Use clinical relationships between the diagnosis and procedure codes (e.g., CPT codes 99213–99215 align with ICD-10 codes related to outpatient management of infections).
        4. Consider common practices, severity, and justifications required in a clinical setting.
        5.Explain possible scenarios where the procedure would or would not apply.
        6. Use common sense, if the diagnosis and procedure codes do not match then throw Error but please for things like fever not being outpatient or inpatient, do not say Error.
        7. If there is a potential mismatch, say Error, and explain the error, use a red cross ✗ and clearly explain the issue and the solution in simple terms. But please avoid long answers. Keep it clear.
        
        8. If there is no huge mistmatch, like typhoid fever and short stay room or services that could be relevant, please do not output Error. Like for fever, it could be in patient or out patinet, it depends.
        9. If the difference is non-debatable, only then show Error like dental services for fever or medical devices for vision, surgery codes for Out patient. If there is no clear mismatch, please just say All Checks passed.
        10. Avoid adding extra technical jargon; use simple wording that a claims adjuster or insurance professional can understand.
        12. Please mention to the user your percentage of confidence in the answer if the answer is not clear for ICD-10 and CPT code mismtach as well.
        
        13.Let me give you an example of typhoid fever or fever as it is not necessarily an out patient condition so do not say Error and use red cross. Just mention all checks passed.
        14.The diagnosis of typhoid fever can warrant inpatient care, including room and board and consultations, especially in moderate to severe cases requiring close monitoring and intravenous treatment.
        
        
        RULE: Fever can be managed in both outpatient and inpatient settings, depending on the severity of the case. Moderate to severe cases often require inpatient care for intravenous antibiotics and monitoring.
        



Here is the input data:
Diagnosis Description (ICD-10): {diagnosis_description}
Procedure Description (CPT): {procedure_description}

Analyze the input and provide your output only when there is an Error.
""",
        input_variables=["diagnosis_description", "procedure_description"]
    )

    # Generate the prompt
    prompt = prompt_template.format(
        diagnosis_description=diagnosis_description,
        procedure_description=procedure_description
    )

    # Invoke the ChatGroq model
    print("Sending request to ChatGroq for coding consistency analysis...")
    response = chat.invoke(prompt)
    print("Response received from ChatGroq.")
    # Remove <think> tags and everything in between them
        
    # Extract the content attribute
    raw_content = response.content
    
    # Remove <think> tags and everything in between them
    cleaned_response = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
    # Return the HTML response
    return cleaned_response



###########################################
# Use case: Claim Type to be mapped with Service Type:
def check_claim_service_type_consistency(sections):
    # Create mapping dictionary with full descriptions
    # Get claim type from the JSON
    claim_type = sections['entry'][1]['resource']['meta']['profile'][0]
    claim_types_mapping = {
    "http://nphies.sa/fhir/ksa/nphies-fs/StructureDefinition/vision-claim|1.0.0": "Vision claims for professional services and products such as glasses and contact lenses",
    "http://nphies.sa/fhir/ksa/nphies-fs/StructureDefinition/institutional-claim|1.0.0": "Hospital, clinic inpatient claims", 
    "http://nphies.sa/fhir/ksa/nphies-fs/StructureDefinition/oral-claim|1.0.0": "Dental, Denture and Hygiene claims",
    "http://nphies.sa/fhir/ksa/nphies-fs/StructureDefinition/pharmacy-claim|1.0.0": "Pharmacy claims for goods and services",
    "http://nphies.sa/fhir/ksa/nphies-fs/StructureDefinition/professional-claim|1.0.0": "Outpatient claims from Physician, Psychological, Chiropractor, Physiotherapy, Speech Pathology, rehabilitative, consultation"
    }

    prodecural_codes_mapping = {
        "http://nphies.sa/terminology/CodeSystem/transportation-srca": "This code set includes Ambulance and transportation services (SRCA)",
        "http://nphies.sa/terminology/CodeSystem/imaging": "This code set includes Imaging Procedures",
        "http://nphies.sa/terminology/CodeSystem/laboratory": "This code set includes Laboratory tests, observations and Blood Bank products",
        "http://nphies.sa/terminology/CodeSystem/medical-devices": "This code set includes Medical devices",
        "http://nphies.sa/terminology/CodeSystem/oral-health-ip": "This code set includes Oral Health - In-patient",
        "http://nphies.sa/terminology/CodeSystem/oral-health-op": "This code set includes Oral Health - Out-patient",
        "http://nphies.sa/terminology/CodeSystem/medication-codes": "This code set includes all drug or medicament substance codes and all pharmaceutical products",
        "http://nphies.sa/terminology/CodeSystem/procedures": "This code set includes Procedures / Health interventions",
        "http://nphies.sa/terminology/CodeSystem/services": "This code set includes Room and Board, In-patient Rounding, Consultations, Services",
        "http://nphies.sa/terminology/CodeSystem/body-site": "This code set includes Specific and identified anatomical location of the service provided to the patient (limb, tooth, etc.)"
    }


    # Extract the link from sections
    link = sections['ProductService'][0]['productOrService']['coding'][0]['system'] 
    service_type = prodecural_codes_mapping.get(link, "Explanation not found for this link.")
    
    claim_type = claim_types_mapping.get(claim_type, "Explanation not found for this link.")
    # use model to check consistency and add to final output.....to be continued later
    # Get the explanation from the mapping

    return None








# Sample examples include Dental, Medical devices cannot be used in vision claim, surgery codes for example you are relating them to OPD. # Core Workflow Structure and Function Purposes

"""
1. Main Entry Points:
- `/chat`: Handles general chat queries
- `/analyze-fhir`: Handles FHIR message analysis

2. Core Processing Pipeline:

# Data Extraction and Organization
def extract_sections(data):
    # Takes raw FHIR data and organizes it into logical sections (Claim, CareTeam, etc.)
    # Creates a structured dictionary for easier validation and analysis
    # Returns: Organized sections dictionary

# Section Validation
def validate_section(section, required_keys, section_name=None):
    # Validates individual sections based on required keys
    # Handles general validation for most sections
    # Returns: Validation result string

def validate_supporting_info(supporting_info):
    # Specialized validation for SupportingInformationSection
    # Implements sequence-specific validation rules
    # Returns: Validation result string

# Code Value Lookup
def fetch_code_value_and_display_value(sections):
    # Retrieves ICD-10 code details from database
    # Returns: Formatted string with code details

# Procedural Code Explanation
def get_procedural_code_explanation(sections):
    # Fetches description for procedural codes (CPT)
    # Returns: Explanation string

# Coding Consistency Check
def check_coding_consistency(diagnosis_description, procedure_description):
    # Uses ChatGroq to analyze alignment between diagnosis (ICD-10) and procedure (CPT)
    # Returns: HTML-formatted analysis result

# Claim Section Analysis
def check_claim_section(claim_section_data):
    # Final validation of claim submission
    # Generates HTML-formatted validation report
    # Returns: HTML string with validation results

# Main Analysis Workflow
def analyze_fhir_endpoint(request: Request):
    # Main entry point for FHIR analysis
    # Workflow:
    # 1. Extract sections from raw FHIR data
    # 2. Validate each section
    # 3. Fetch code values
    # 4. Get procedural explanations
    # 5. Check coding consistency
    # 6. Generate final claim validation report
    # Returns: JSON response with analysis results
"""