from flask import Flask, request, jsonify # type: ignore
from flask_cors import CORS
import os
import base64 
import json 
from werkzeug.utils import secure_filename
from google import genai
from google.genai import types

# --- 0. CONFIGURATION & MOCK DATA STORE ---
# NOTE: The API key is now loaded from the environment variable GEMINI_API_KEY
# If the environment variable is not set, it falls back to the hardcoded value below for testing.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBDpYpE-q3GV2C-By4k3_dFrGFfZAcu9NU")

PANTRY = [
    {"name": "Milk (2%)", "status": "expiring_soon", "expiry_days": 3},
    {"name": "Flour (All-purpose)", "status": "ok", "expiry_days": 180},
    {"name": "Eggs", "status": "ok", "expiry_days": 14},
]

GROCERY_LIST = [
    "Butter",
    "Tomato Sauce",
]

# --- 1. MIME TYPE UTILITY (FIX FOR SCANNING ISSUE) ---

def _get_mime_type_from_filepath(filepath: str) -> str:
    """Infers the MIME type from the file extension for Gemini Vision API."""
    filename = os.path.basename(filepath)
    extension = filename.rsplit('.', 1)[-1].lower()
    
    if extension in ('jpg', 'jpeg'):
        return 'image/jpeg'
    elif extension == 'png':
        return 'image/png'
    elif extension == 'webp':
        return 'image/webp'
    return 'application/octet-stream' # Default or unknown

# --- 2. DATA MANAGEMENT FUNCTIONS ---

def get_current_inventory():
    """Returns the current pantry inventory data."""
    return PANTRY

def get_grocery_list():
    """Returns the current grocery list."""
    return GROCERY_LIST

def add_to_pantry(item_name: str):
    """Adds a new item to the pantry (used by OCR confirmed items)."""
    # Simple check to prevent immediate duplicates, assuming fresh item
    if not any(item['name'] == item_name for item in PANTRY):
        PANTRY.append({"name": item_name, "status": "ok", "expiry_days": 90})
        return True
    return False

def add_to_grocery_list(item_name: str) -> bool:
    """Adds an item to the grocery list (used by the Gemini tool)."""
    normalized_name = item_name.strip().lower()
    if normalized_name not in [item.lower() for item in GROCERY_LIST]:
        GROCERY_LIST.append(item_name)
        return True
    return False

def find_item_in_pantry(item_name: str) -> str:
    """Checks the current pantry inventory for a specific item (used by the Gemini tool)."""
    normalized_name = item_name.lower()
    found_items = [
        item for item in PANTRY 
        if normalized_name in item['name'].lower()
    ]
    
    if not found_items:
        return f"The pantry does not currently contain '{item_name}'. You might need to buy it."
    
    # Return a concise summary for the model
    summaries = [f"{item['name']} (Status: {item['status']})" for item in found_items]
    return f"In the pantry: {', '.join(summaries)}. Use this to inform your conversational response."

# --- 3. LIVE OCR FUNCTION (Using Gemini Vision) ---

def scan_receipt_with_gemini(filepath: str):
    """
    Scans a receipt using the Gemini 2.5 Flash model with structured output.
    This replaces the mock function with a real API call.
    """
    global gemini_client # Access the initialized client

    if not gemini_client:
        print("ERROR: Gemini client is not available for receipt scanning.")
        return None

    try:
        # 1. Read the image file and encode it to base64
        with open(filepath, "rb") as f:
            image_data = f.read()

        # FIX: Dynamically determine the MIME type based on file extension
        mime_type = _get_mime_type_from_filepath(filepath)
        
        # 2. Define the structured output schema
        schema = {
            "type": "ARRAY",
            "description": "A list of food items and their prices extracted from the receipt.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING", "description": "The name of the food item, e.g., 'Whole Milk' or 'Apples'."},
                    "price": {"type": "NUMBER", "description": "The price of the item as a floating point number (e.g., 4.99)."},
                },
                "required": ["name", "price"]
            }
        }
        
        # 3. Define the prompt and image parts
        parts = [
            types.Part.from_text("Analyze this grocery store receipt. Extract all purchased food and beverage items and their total price. Do NOT include non-food items like tax, discounts, or services. Return the results as a JSON array matching the provided schema."),
            # Use the determined mime_type for correct image processing
            types.Part.from_bytes(data=image_data, mime_type=mime_type), 
        ]
        
        # 4. Call the Gemini API for structured generation
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=parts,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            )
        )
        
        # 5. Parse the JSON response
        if response.text:
            # The response text is a JSON string matching the schema
            extracted_data = json.loads(response.text)
            print(f"Successfully extracted {len(extracted_data)} items from receipt.")
            return extracted_data
        
        return None

    except Exception as e:
        print(f"ERROR during receipt scanning with Gemini: {e}")
        # Fallback in case of API or parsing error
        return None 

# --- 4. FLASK APP SETUP ---

app = Flask(__name__)
# IMPORTANT: Updated CORS to accept port 5001 for consistency
CORS(app, resources={r"/api/*": {"origins": ["http://127.0.0.1:5001", "http://localhost:5001", "null"]}})

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- 5. CHATBOT CONFIGURATION ---

USER_ALLERGIES = ["Lactose", "Peanuts"] 
ALLERGY_INFO = f"The user is strictly allergic to: {', '.join(USER_ALLERGIES)}."

SYSTEM_PROMPT = (
    "You are the 'Pantry Twin', a helpful, friendly, and smart kitchen assistant. "
    "Your goal is to answer questions conversationally and accurately. "
    "You can use your general knowledge, but you MUST prioritize the user's PANTRY SUMMARY and ALLERGIES when suggesting recipes or food items. "
    "Use the available tools (add_to_grocery_list, find_item_in_pantry) when appropriate. "
    "**MODE 2: GROCERY LIST BUILDING.** If the user asks for a dish or needs an item that requires items not in the pantry, you MUST suggest 2-3 key ingredients they would need to buy. **To suggest buying an item, you MUST call the `add_to_grocery_list` tool.** "
    "You can answer general questions about food, recipes, or inventory without using a tool command. "
    f"IMPORTANT: {ALLERGY_INFO}"
)

# Define the tools available to the Gemini conversational model
tools = [add_to_grocery_list, find_item_in_pantry] 

# Global variable to store the chat session
gemini_client = None
chat_session = None
CONVERSATIONAL_MODEL = 'gemini-2.5-flash'


def initialize_chatbot():
    """Initializes the Gemini client and starts a new chat session."""
    global gemini_client, chat_session
    
    try:
        if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_API_KEY_HERE":
            # Initialize client with the API key
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            print("INFO: Gemini Client initialized.")
        else:
            # Initialize client without API key (will use environment variable or fail)
            gemini_client = genai.Client() 
            print("INFO: Gemini Client initialized (using environment variable or hardcoded fallback).")

        # Tools must be passed inside the config object when using chats.create
        chat_session = gemini_client.chats.create(
            model=CONVERSATIONAL_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7,
                tools=tools
            ),
        )
        print("INFO: New persistent chat session created.")
        
    except Exception as e:
        print(f"ERROR: Gemini Client or Chat Session failed to initialize. Error: {e}")
        import traceback
        traceback.print_exc()
        gemini_client = None
        chat_session = None

        
# --- 6. UTILITY FUNCTIONS ---
def allowed_file(filename):
    """Checks if the uploaded file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- 7. API ENDPOINTS ---

@app.route('/api/pantry', methods=['GET'])
def get_pantry():
    return jsonify(get_current_inventory())

@app.route('/api/grocery_list', methods=['GET'])
def get_grocery_list_endpoint():
    return jsonify(get_grocery_list())
    
@app.route('/api/reset_chat', methods=['POST'])
def reset_chat_session():
    """NEW ENDPOINT: Resets the current persistent chat session, effectively clearing the history."""
    # Re-initialize the chatbot, which creates a brand new chat_session
    initialize_chatbot() 
    if chat_session:
        return jsonify({"status": "success", "message": "Chat session reset successfully."})
    else:
        return jsonify({"status": "error", "message": "Failed to initialize new chat session."}), 500


@app.route('/api/add_scanned_items', methods=['POST'])
def add_scanned_items():
    data = request.get_json()
    if not data or 'items' not in data:
        return jsonify({"status": "error", "message": "Invalid data format. 'items' list required."}), 400

    items_to_add = data['items']
    added_count = 0
    
    for item in items_to_add:
        # Ensure the item object has a 'name' key before trying to add it
        if 'name' in item and add_to_pantry(item['name']):
            added_count += 1
            
    return jsonify({
        "status": "success", 
        "message": f"Successfully added {added_count} items to the pantry.",
        "added_count": added_count
    })

@app.route('/api/scan_receipt_live', methods=['POST'])
def scan_receipt_live():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "No selected file or file type not allowed"}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(filepath)
        
        try:
            # Call the LIVE OCR utility function
            extracted_items = scan_receipt_with_gemini(filepath)
            
            if extracted_items is not None:
                # The frontend expects a list of items with name and price
                return jsonify({"status": "success", "items": extracted_items})
            else:
                return jsonify({"status": "error", "message": "Failed to extract items. Check console for errors."}), 500
        
        finally:
            # Clean up the file after processing
            if os.path.exists(filepath):
                 os.remove(filepath)

@app.route('/api/chat_history', methods=['GET'])
def get_chat_history():
    """
    Retrieves the current conversation history from the Gemini chat session.
    """
    global chat_session
    
    if not chat_session:
        return jsonify({"status": "error", "message": "Chatbot not initialized."}), 503
        
    try:
        history = chat_session.get_history()
        # Format the history for the frontend
        formatted_history = []
        for message in history:
            # Filter for text parts and ignore tool-related parts when showing history
            if message.role in ["user", "model"] and message.parts:
                # Find the first part that contains text
                text_part = next((p for p in message.parts if p.text), None)
                if text_part:
                    formatted_history.append({
                        "role": message.role,
                        # Clean up any residual context from the beginning of the user message
                        "text": text_part.text.split("--- USER QUESTION ---\n")[-1]
                    })
        
        return jsonify({"status": "success", "history": formatted_history})
    except Exception as e:
        print(f"ERROR retrieving chat history: {e}")
        return jsonify({"status": "error", "message": "Failed to retrieve history."}), 500


@app.route('/api/converse', methods=['POST'])
def converse():
    """Handles conversational requests with the Gemini Pantry Twin, including tool use."""
    global chat_session

    if not chat_session:
        # Re-initialize the chat session if it was lost (e.g., due to a server restart)
        initialize_chatbot()
        if not chat_session:
             return jsonify({"status": "error", "message": "Gemini Chatbot is not initialized (API Key missing or invalid)."}), 503

    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"status": "error", "message": "No question provided."}), 400
    
    # Get current pantry and grocery list data to inject into the prompt
    pantry_summary = "Current Pantry Items: " + ", ".join([item['name'] for item in get_current_inventory()])
    list_summary = "Current Grocery List: " + ", ".join(get_grocery_list())
    
    # Combine context and user question
    # IMPORTANT: We prepend the context to the user message on every turn
    full_prompt = (
        f"--- CONTEXT ---\n"
        f"Pantry: {pantry_summary}\n"
        f"List: {list_summary}\n"
        f"--- USER QUESTION ---\n{question}"
    )
    
    try:
        # 1. First call: Send user prompt to the persistent chat session
        response = chat_session.send_message(full_prompt)

        # Check if the model decided to call a function/tool
        if response.function_calls:
            tool_responses = []
            func_args = {}
            
            for call in response.function_calls:
                func_name = call.name
                # Map tool names back to actual functions in this script
                if func_name == 'add_to_grocery_list':
                    func = add_to_grocery_list
                elif func_name == 'find_item_in_pantry':
                    func = find_item_in_pantry
                else:
                    func = None

                if func:
                    func_args = dict(call.args)
                    result = func(**func_args)
                    
                    # This is the correct way to handle function responses for the Gemini API
                    tool_responses.append(
                        types.Content(
                            role="tool",
                            parts=[types.Part.from_function_response(
                                name=func_name, 
                                response={"result": result}
                            )]
                        )
                    )
            
            # 2. Second call: Send the tool output back to the persistent chat session
            final_response = chat_session.send_message(
                contents=tool_responses
            )
            
            # Determine if the action was adding to the grocery list
            is_list_action = any(call.name == "add_to_grocery_list" for call in response.function_calls)
            item_name_used = func_args.get('item_name', 'Unknown Item') if response.function_calls else 'Unknown Item'

            if is_list_action:
                return jsonify({
                    "status": "action_success",
                    "action": "ADD_TO_LIST",
                    "message": final_response.text,
                    "item": item_name_used
                })
            else:
                return jsonify({"status": "success", "response": final_response.text})

        else:
            # Standard conversational response (no tool use)
            return jsonify({"status": "success", "response": response.text})

    except Exception as e:
        print(f"An unexpected error occurred during conversation: {e}")
        return jsonify({"status": "error", "message": f"Server error: {e}"}), 500

# Default route for the root URL
@app.route('/')
def index():
    # Explicitly directing user to port 5001 now.
    return "Smart Grocery Navigator Backend Running. Connect to http://127.0.0.1:5001"

if __name__ == '__main__':
    # Initialize the chatbot when the server starts
    initialize_chatbot()
    # FIX: Changed port from 5000 to 5001 to resolve 'Address already in use' error
    print("Starting Flask server on http://127.0.0.1:5001")
    app.run(debug=True, port=5001)
