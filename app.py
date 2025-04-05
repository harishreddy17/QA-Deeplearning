from flask import Flask, render_template, request, jsonify, session
import time
from chat import PorscheBot
import os
import traceback

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = "your-secret-key-here"

# Initialize the bot
try:
    chatbot = PorscheBot()
except Exception as e:
    print(f"Error initializing chatbot: {str(e)}")
    print(traceback.format_exc())
    chatbot = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    try:
        if not chatbot:
            return jsonify({
                "response": "I apologize, but the chatbot is currently unavailable. Please try again later.",
                "suggestions": [],
                "state": None
            })

        user_message = request.json.get("message", "").strip()
        state = request.json.get("state", None)
        
        print(f"Received state: {state}")  # Debug log
        
        if not user_message:
            return jsonify({
                "response": "I didn't receive your message. Please try again.",
                "suggestions": [],
                "state": state
            })

        # Get response from the chatbot
        result = chatbot.get_response(user_message, state)
        
        print(f"Chatbot response state: {result.get('state')}")  # Debug log
        
        # Ensure the response has the correct structure
        if not isinstance(result, dict):
            result = {
                "response": str(result),
                "suggestions": [],
                "state": state
            }
        
        if "suggestions" not in result:
            result["suggestions"] = []
            
        if "state" not in result:
            result["state"] = state

        return jsonify(result)

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "response": "I apologize, but I encountered an error processing your request. Please try again.",
            "suggestions": [],
            "state": state
        })

@app.route("/set_name", methods=["POST"])
def set_name():
    name = request.json.get("name", "")
    if name:
        session["user_name"] = name
        return jsonify({"success": True})
    return jsonify({"success": False})

if __name__ == "__main__":
    app.run(debug=True)
