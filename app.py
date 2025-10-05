# app.py

# --- IMPORTS ---
from flask import Flask, request, jsonify, render_template
from datetime import datetime, timedelta
import os
import sys

from google import genai 
from google.genai.errors import APIError

# --- CONFIGURATION & INITIALIZATION ---

# SECURITY BEST PRACTICE: Use os.getenv() to load the key securely.
# Ensure you set the GEMINI_API_KEY environment variable before running.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_HARDCODED_KEY_HERE_FOR_TESTING_ONLY") 
# Note: The hardcoded key is left here for quick testing, but using os.getenv is highly recommended for production.

app = Flask(__name__)
# Initialize the Gemini Client globally
try:
    if not GEMINI_API_KEY:
        print("FATAL ERROR: GEMINI_API_KEY is not set.")
        sys.exit()
    client = genai.Client(api_key=GEMINI_API_KEY) 
except Exception as e:
    print(f"FATAL ERROR: Could not initialize Gemini client. Error: {e}")
    sys.exit()

# --- PANTRY DATA & CONSTANTS ---
PANTRY_INVENTORY = []
USER_GROCERY_LIST = [] 
USER_ALLERGIES = ["Lactose", "Peanuts"] 

DEFAULT_EXPIRY = {
    "Milk": 7, "Bread": 5, "Eggs": 21, "Apples": 14, 
    "Pasta": 365, "Cheese": 14
}


# ----------------------------------------------------
# --- BACKEND FUNCTIONS (Modified to run standalone) ---
# ----------------------------------------------------

# Note: The original scan_receipt is run ONCE at startup to populate the pantry
def scan_receipt(image_path="receipt.jpg"):
    # ... (Your existing receipt logic remains the same) ...
    return [
        {"name": "Milk", "price": 4.50},
        {"name": "Bread", "price": 3.00},
        {"name": "Cheese", "price": 6.00},
        {"name": "Eggs", "price": 3.50},
        {"name": "Pasta", "price": 2.25}
    ]

def add_to_pantry(items_list):
    """Adds scanned items to the main inventory list."""
    for item in items_list:
        item_name = item['name'].title()
        days_to_expire = DEFAULT_EXPIRY.get(item_name, 30)
        expiry_date = datetime.now() + timedelta(days=days_to_expire)
        pantry_record = {
            "name": item_name,
            "purchase_date": datetime.now().strftime("%Y-%m-%d"),
            "expiry_date": expiry_date.strftime("%Y-%m-%d"),
            "is_alerted": False,
            "allergens": ["Lactose"] if item_name == "Milk" else []
        }
        PANTRY_INVENTORY.append(pantry_record)

# The core function used by the AI to update the grocery list
def add_to_grocery_list(item_name):
    """Allows the user to manually add an item to the list."""
    item_name = item_name.strip().title()
    if item_name not in USER_GROCERY_LIST:
        USER_GROCERY_LIST.append(item_name)
        # We don't print, we just return the status
        return {"status": "success", "item": item_name}
    return {"status": "duplicate", "item": item_name}

# Generates the smart list for the frontend to display
def generate_smart_list():
    """Generates the suggestions list."""
    final_list_items = set() 
    today = datetime.now().date()
    
    # 1. Add expiring items (Do Not Buy)
    for item in PANTRY_INVENTORY:
        item_expiry = datetime.strptime(item['expiry_date'], "%Y-%m-%d").date()
        days_left = (item_expiry - today).days
        if item['name'] == 'Bread': days_left = 1 # Demo logic
        
        if days_left <= 3 and days_left >= 0:
            final_list_items.add(f"DO NOT BUY: {item['name']} ({days_left} days left)")

    # 2. Add missing common items (Buy)
    current_items = [item['name'] for item in PANTRY_INVENTORY]
    common_items = ["Milk", "Bread", "Eggs", "Cheese", "Water"]
    for item in common_items:
        if item not in current_items:
            final_list_items.add(f"BUY: {item} (Missing)")
            
    # 3. Add user-requested items
    for item in USER_GROCERY_LIST:
         final_list_items.add(f"USER REQUEST: {item}")

    return sorted(list(final_list_items))


# Refactored to RETURN the AI's result as a dictionary (JSON response)
def ask_pantry_twin(user_question, current_pantry_items, allergies):
    """
    Sends a user question and the pantry contents to the Gemini model
    and returns a JSON response ready dictionary.
    """
    # 1. Format the Pantry Data
    pantry_summary = "Current Pantry Items and Expiry Dates:\n"
    for item in current_pantry_items:
        pantry_summary += f"- {item['name']} (Expires: {item['expiry_date']})\n"
    
    manual_list_summary = f"Current User-Added List: {', '.join(USER_GROCERY_LIST)}"
    allergy_info = f"The user is strictly allergic to: {', '.join(allergies)}."
    
    # 2. Create the System Prompt (Unchanged)
    system_prompt = (
        "You are the 'Pantry Twin', a helpful, friendly, and smart kitchen assistant. "
        "Your goal is to answer questions based ONLY on the provided pantry inventory and the user's known allergies. "
        "Always be conversational, but keep your answers concise. "
        "If suggesting a recipe, ONLY use ingredients listed in the PANTRY SUMMARY. "
        "You can answer questions about the current inventory or suggest recipes. "
        "**Crucially, if the user asks you to add an item to their grocery list, you MUST only respond with the exact phrase: ** 'ADD_TO_LIST: [Item Name]' **and nothing else.** "
        f"IMPORTANT: {allergy_info}"
    )
    
    # 3. Combine all the information into the prompt 
    full_prompt = (
        f"--- PANTRY SUMMARY ---\n{pantry_summary}\n"
        f"--- USER-ADDED LIST ---\n{manual_list_summary}\n"
        f"--- USER QUESTION ---\n{user_question}"
    )

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0
            )
        )
        ai_response = response.text.strip()
        
        # --- Handle "Tool Use" Logic ---
        if ai_response.startswith('ADD_TO_LIST:'):
            item_to_add = ai_response.split(':', 1)[1].strip()
            add_result = add_to_grocery_list(item_to_add)
            
            # Return the action status to the frontend
            return {"status": "action_success", "action": "ADD_TO_LIST", "item": item_to_add, "message": f"Successfully added {item_to_add} to your list."}
        else:
            # Return the conversational text
            return {"status": "success", "response": ai_response}

    except APIError as e:
        return {"status": "error", "message": f"Gemini API Error: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}


# ----------------------------------------------------
# --- FLASK API ROUTES ---
# ----------------------------------------------------

@app.route('/')
def home():
    """Serves the main HTML page (must be in the 'templates' folder)."""
    return render_template('index.html')

@app.route('/api/pantry', methods=['GET'])
def get_pantry():
    """Returns the full pantry inventory for display."""
    # We must run the expiry alert check here to update the 'days_left' status
    
    # A cleaner way to calculate days left for the frontend
    today = datetime.now().date()
    pantry_status = []
    for item in PANTRY_INVENTORY:
        item_expiry = datetime.strptime(item['expiry_date'], "%Y-%m-%d").date()
        days_left = (item_expiry - today).days
        
        # Apply the demo logic for Bread
        if item['name'] == 'Bread': days_left = 1 
        
        pantry_status.append({
            "name": item['name'],
            "expiry_date": item['expiry_date'],
            "days_left": days_left,
            "is_alert": days_left <= 3 and days_left >= 0,
            "is_expired": days_left < 0,
            "allergens": item['allergens']
        })
    
    return jsonify(pantry_status)

@app.route('/api/grocery_list', methods=['GET'])
def get_grocery_list():
    """Returns the smart shopping list for display."""
    smart_list = generate_smart_list()
    return jsonify(smart_list)

@app.route('/api/converse', methods=['POST'])
def handle_conversation():
    """The main endpoint for the conversational AI interaction."""
    data = request.json
    user_question = data.get('question', '')
    
    if not user_question:
        return jsonify({"status": "error", "message": "No question provided."}), 400
    
    # Call the refactored AI function
    result = ask_pantry_twin(user_question, PANTRY_INVENTORY, USER_ALLERGIES)
    
    return jsonify(result)

# ----------------------------------------------------
# --- APP SETUP ---
# ----------------------------------------------------

# Initial setup function runs once when the Flask app starts
def initial_pantry_setup():
    """Populates the pantry when the server starts."""
    items_bought = scan_receipt() 
    add_to_pantry(items_bought)

if __name__ == '__main__':
    print("--- STARTING PANTRY TWIN WEB SERVER ---")
    initial_pantry_setup()
    
    # Check for alerts at startup (will print to server console)
    # The frontend uses the /api/pantry route to get the alert status
    check_for_alerts(my_allergies=USER_ALLERGIES) 
    
    print(f"🤖 Pantry initialized with {len(PANTRY_INVENTORY)} items.")
    print("🌐 Server running on http://127.0.0.1:5000")
    
    app.run(debug=True)