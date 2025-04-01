from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import requests
import base64
import json
from flask_cors import CORS
from flask import Flask, render_template

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get API keys from environment variables
PLANT_ID_API_KEY = os.getenv('PLANT_ID_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

conversation_contexts = {}

@app.route('/api/identify-plant', methods=['POST'])
def identify_plant():
    try:
        # Get data from request
        data = request.json
        image_base64 = data.get('image')
        session_id = data.get('session_id', 'default')
        # latitude = data.get('latitude')
        # longitude = data.get('longitude')
        # similar_images = data.get('similar_images', True)
        
        if session_id not in conversation_contexts:
            conversation_contexts[session_id] = []
        
        request_body = {
            'images': [image_base64]
        }
        
        # if latitude and longitude:
        #     request_body['latitude'] = float(latitude)
        
        #     request_body['longitude'] = float(longitude)
        
        # if similar_images:
        #     request_body['similar_images'] = similar_images
        
        # Call Plant.id API
        response = requests.post(
            'https://plant.id/api/v3/identification',
            headers={
                'Content-Type': 'application/json',
                'Api-Key': PLANT_ID_API_KEY
            },
            json=request_body
        )
        
        response.raise_for_status()
        plant_data = response.json()
    
        conversation_contexts[session_id].append({
            'role': 'user',
            'parts': [{'text': "I've uploaded a plant image for identification"}]
        })
        
        conversation_contexts[session_id].append({
            'role': 'model',
            'parts': [{'text': "Analyzing your plant image... Please wait."}]
        })
        
        if (plant_data and plant_data.get('result') and 
            plant_data.get('result').get('classification') and 
            plant_data.get('result').get('classification').get('suggestions') and 
            len(plant_data.get('result').get('classification').get('suggestions')) > 0):
            
            top_result = plant_data['result']['classification']['suggestions'][0]
            plant_name = top_result['name']
            common_names = top_result.get('details', {}).get('common_names', [])
            common_names_str = ", ".join(common_names) if common_names else ""
            description = top_result.get('details', {}).get('description', {}).get('value', "")
            
            # Call Gemini API for a welcome message
            gemini_prompt = f"I just identified a plant as {plant_name}. "
            if common_names_str:
                gemini_prompt += f"It's also known as: {common_names_str}. "
            if description:
                gemini_prompt += f"Description: {description} "
            gemini_prompt += "Can you provide a short welcome message acknowledging that you know about this plant?"
            
            gemini_response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
                headers={'Content-Type': 'application/json'},
                json={
                    'contents': [
                        {
                            'role': 'user',
                            'parts': [{'text': gemini_prompt}]
                        }
                    ]
                }
            )
            
            gemini_response.raise_for_status()
            gemini_data = gemini_response.json()
            
            welcome_message = ""
            if (gemini_data.get('candidates') and len(gemini_data['candidates']) > 0 and 
                gemini_data['candidates'][0].get('content') and 
                gemini_data['candidates'][0]['content'].get('parts') and 
                len(gemini_data['candidates'][0]['content']['parts']) > 0):
                
                welcome_message = gemini_data['candidates'][0]['content']['parts'][0]['text']
                
                # Add welcome message to conversation context
                conversation_contexts[session_id].append({
                    'role': 'model',
                    'parts': [{'text': welcome_message}]
                })
            
            # Add identified plant info to the response
            identified_info = f"I've identified this plant as {plant_name}. "
            if common_names_str:
                identified_info += f"Also known as: {common_names_str} "
            identified_info += "You can now ask me questions about this plant!"
            
            conversation_contexts[session_id].append({
                'role': 'model',
                'parts': [{'text': identified_info}]
            })
            
            # Reduce conversation context if it's too long
            if len(conversation_contexts[session_id]) > 10:
                conversation_contexts[session_id] = conversation_contexts[session_id][-10:]
            
            return jsonify({
                'plant_data': plant_data,
                'welcome_message': welcome_message,
                'conversation_context': conversation_contexts[session_id]
            })
        
        # If identification failed
        conversation_contexts[session_id].append({
            'role': 'model',
            'parts': [{'text': "I couldn't identify this plant from the image. Please try with a clearer image or a different plant."}]
        })
        
        return jsonify({
            'plant_data': {},
            'error': "Could not identify the plant",
            'conversation_context': conversation_contexts[session_id]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')
        current_plant = data.get('current_plant', None)
        
        if session_id not in conversation_contexts:
            conversation_contexts[session_id] = []
        
        conversation_contexts[session_id].append({
            'role': 'user',
            'parts': [{'text': message}]
        })
        
        user_query = message
        if current_plant:
            user_query = f"The user is asking about a plant that was identified as {current_plant.get('name')} "
            common_names = current_plant.get('commonNames', [])
            if common_names:
                user_query += f"({', '.join(common_names)}) "
            description = current_plant.get('description')
            if description:
                user_query += f"Description: {description} "
            user_query += f"With this context in mind, please answer the following question: {message}"
        
        # Call Gemini API
        payload = {
            'contents': conversation_contexts[session_id]
        }
        
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
            headers={'Content-Type': 'application/json'},
            json=payload
        )
        
        response.raise_for_status()
        data = response.json()
        
        if (data.get('candidates') and len(data['candidates']) > 0 and 
            data['candidates'][0].get('content') and 
            data['candidates'][0]['content'].get('parts') and 
            len(data['candidates'][0]['content']['parts']) > 0):
            
            bot_response = data['candidates'][0]['content']['parts'][0]['text']
            
            # Add bot response to conversation context
            conversation_contexts[session_id].append({
                'role': 'model',
                'parts': [{'text': bot_response}]
            })
            
            if len(conversation_contexts[session_id]) > 10:
                conversation_contexts[session_id] = conversation_contexts[session_id][-10:]
            
            return jsonify({
                'response': bot_response,
                'conversation_context': conversation_contexts[session_id]
            })
        
        return jsonify({
            'error': "Could not generate a response",
            'conversation_context': conversation_contexts[session_id]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)