import json
import torch
from model import NeuralNet
from utils import tokenize
from advanced_nlp import AdvancedNLP
import random


class PorscheBot:
    def __init__(self):
        try:
            # Device configuration
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")

            # Load intents
            with open('intents.json', 'r') as f:
                self.intents = json.load(f)['intents']
            
            # Load car data
            with open('cars.json', 'r') as f:
                self.car_data = json.load(f)

            # Load pre-trained model data with proper device mapping
            data_dir = "chatdata.pth"
            try:
                data = torch.load(data_dir, map_location=self.device)
            except RuntimeError as e:
                print(f"Error loading model: {str(e)}")
                print("Attempting to load on CPU...")
                data = torch.load(data_dir, map_location=torch.device('cpu'))

            self.input_size = data["input_size"]
            self.hidden_size = data["hidden_size"]
            self.output_size = data["output_size"]
            self.all_words = data["all_words"]
            self.tags = data["tags"]
            self.model_state = data["model_state"]

            # Initialize the model
            self.model = NeuralNet(self.input_size, self.hidden_size, self.output_size).to(
                self.device
            )
            self.model.load_state_dict(self.model_state)
            self.model.eval()
            
            # Initialize NLP
            self.nlp = AdvancedNLP()
            
            # Initialize suggestion templates
            self.suggestion_templates = {
                'general': [
                    "Would you like to know more about {model}'s performance?",
                    "Interested in {model}'s customization options?",
                    "Would you like to schedule a test drive for the {model}?",
                    "Would you like to explore financing options for the {model}?"
                ],
                'performance': [
                    "Would you like to know about {model}'s fuel efficiency?",
                    "Interested in {model}'s transmission options?",
                    "Would you like to compare {model}'s performance with other models?",
                    "Would you like to know about {model}'s warranty coverage?"
                ],
                'customization': [
                    "Would you like to see {model}'s available colors?",
                    "Interested in {model}'s interior package options?",
                    "Would you like to know about {model}'s exclusive customization features?",
                    "Would you like to explore {model}'s performance upgrade options?"
                ],
                'pricing': [
                    "Would you like to know about financing options for the {model}?",
                    "Interested in lease options for the {model}?",
                    "Would you like to know about maintenance packages for the {model}?",
                    "Would you like to schedule a test drive for the {model}?"
                ],
                'features': [
                    "Would you like to know about {model}'s interior features?",
                    "Interested in {model}'s technology features?",
                    "Would you like to know about {model}'s safety features?",
                    "Would you like to explore {model}'s comfort features?"
                ]
            }
            
        except Exception as e:
            print(f"Error initializing PorscheBot: {str(e)}")
            raise

    def get_car_info(self, car_name):
        """
        Fetch details about the car from the car data (from 'cars.json').
        """
        try:
            # First try exact match (case-insensitive)
            for model in self.car_data['models']:
                if car_name.lower() == model['name'].lower():
                    return model
            
            # If no exact match, try partial match with priority to longer names
            matching_models = []
            for model in self.car_data['models']:
                model_name_parts = model['name'].lower().split()
                input_parts = car_name.lower().split()
                
                # Count how many parts of the model name appear in the input
                matching_parts = sum(1 for part in model_name_parts if part in input_parts)
                
                # Require at least 2 matching parts to avoid false positives
                if matching_parts >= 2:
                    matching_models.append((model, matching_parts, len(model_name_parts)))
            
            if matching_models:
                # Sort by number of matching parts (descending) and then by model name length (descending)
                matching_models.sort(key=lambda x: (-x[1], -x[2]))
                return matching_models[0][0]
            
            return None
        except Exception as e:
            print(f"Error getting car info: {str(e)}")
            return None

    def get_sales_suggestions(self, car_model=None, question_type='general'):
        try:
            suggestions = []
            
            # Add model-specific suggestions if a car model is provided
            if car_model:
                # Get templates based on question type
                templates = self.suggestion_templates.get(question_type, [])
                if not templates:
                    templates = self.suggestion_templates['general']
                
                # Add model-specific suggestions
                for template in templates:
                    suggestions.append(template.format(model=car_model))
            
            # Add general sales suggestions
            general_suggestions = [
                "Would you like to visit one of our dealerships?",
                "Would you like to know about our maintenance packages?",
                "Would you like to explore our financing options?",
                "Would you like to schedule a test drive?"
            ]
            
            # Combine and randomize suggestions
            all_suggestions = suggestions + general_suggestions
            random.shuffle(all_suggestions)
            
            # Return top 4 unique suggestions
            return list(dict.fromkeys(all_suggestions))[:4]
            
        except Exception as e:
            print(f"Error generating suggestions: {str(e)}")
            return []

    def get_response(self, user_message):
        try:
            # Check if user is asking about available models
            if any(phrase in user_message.lower() for phrase in ['what models', 'which models', 'available models', 'car models', 'lineup']):
                # Get all model names
                model_names = [model['name'] for model in self.car_data['models']]
                response = f"We offer the following Porsche models: {', '.join(model_names)}. Would you like to know more about any specific model? Visit 'https://www.porsche.com/usa/models/'"
                suggestions = [f"Tell me about the {model}" for model in model_names[:4]]
                return {
                    'response': response,
                    'suggestions': suggestions
                }
            
            # Check if user is asking about the most popular model
            if any(phrase in user_message.lower() for phrase in ['most popular', 'popular model', 'best selling', 'favorite model']):
                # Get the 911 Carrera as the most popular model
                car_info = self.get_car_info('911 Carrera')
                if car_info:
                    response = f"""Our most popular model is the iconic 911 Carrera. This legendary sports car combines timeless design with cutting-edge technology:

• Performance: {car_info['features']['power_PS']} PS ({car_info['features']['power_kW']} kW) of power
• Acceleration: 0-60 mph in {car_info['features']['0-60 mph']}
• Top Speed: {car_info['features']['Top Speed']}
• Engine: {car_info['features']['cylinders']}-cylinder boxer engine
• Transmission: {car_info['features']['transmission']}

Discover the perfect 911 Carrera for you. Visit 'https://www.porsche.com/usa/models/' """
                    suggestions = [
                        "What's the price of the 911 Carrera?",
                        "Tell me about the 911 Carrera's customization options",
                        "Compare the 911 Carrera with other models",
                        "Schedule a test drive for the 911 Carrera"
                    ]
                    return {
                        'response': response,
                        'suggestions': suggestions
                    }
            
            # Analyze user message
            entities = self.nlp.extract_entities(user_message)
            sentiment = self.nlp.analyze_sentiment(user_message)
            question_type = self.nlp.classify_question(user_message)
            
            # Initialize response components
            response = ""
            car_model = None
            suggestions = []
            
            # Check for car model in entities
            if 'car_model' in entities:
                car_model = entities['car_model']
                car_info = self.get_car_info(car_model)
                
                if car_info:
                    # Check for top speed query
                    if any(word in user_message.lower() for word in ['top speed', 'maximum speed', 'max speed', 'how fast']):
                        response = f"The {car_model} has a top speed of {car_info['features']['Top Speed']}."
                    # Check for power query
                    elif any(word in user_message.lower() for word in ['power', 'horsepower', 'hp', 'ps', 'kw', 'engine power']):
                        response = f"The {car_model} has {car_info['features']['power_PS']} PS ({car_info['features']['power_kW']} kW) of power."
                    # Check for acceleration query
                    elif any(word in user_message.lower() for word in ['0-60', 'zero to sixty', 'acceleration', 'how fast']):
                        response = f"The {car_model} accelerates from 0-60 mph in {car_info['features']['0-60 mph']}."
                    # Check for color query
                    elif any(word in user_message.lower() for word in ['color', 'colour', 'paint', 'exterior']):
                        if 'available_colors' in car_info['features']:
                            colors = car_info['features']['available_colors']
                            if isinstance(colors, list):
                                response = f"The {car_model} is available in the following colors: {', '.join(colors)}."
                            else:
                                response = f"The {car_model} is available in {colors}."
                        else:
                            response = f"I don't have specific color information for the {car_model}. Would you like to know about its performance or other features?"
                    # Check for output per liter query
                    elif any(word in user_message.lower() for word in ['output per liter', 'power per liter', 'specific output']):
                        response = f"The {car_model} has a maximum output of {car_info['features']['max_output_per_liter_kW']} kW ({car_info['features']['max_output_per_liter_PS']} PS) per liter."
                    # Check for engine speed query
                    elif any(word in user_message.lower() for word in ['engine speed', 'rpm', 'revolution', 'max engine speed']):
                        response = f"The {car_model} has a maximum engine speed of {car_info['features']['max_engine_speed']} rpm."
                    # Check for stroke query
                    elif any(word in user_message.lower() for word in ['stroke', 'piston stroke']):
                        response = f"The {car_model} has a stroke of {car_info['features']['stroke']}."
                    # Check for torque query
                    elif any(word in user_message.lower() for word in ['torque', 'nm', 'newton meters']):
                        response = f"The {car_model} produces {car_info['features']['max_torque']} of torque."
                    # Check for bore size query
                    elif any(word in user_message.lower() for word in ['bore', 'bore size', 'cylinder bore']):
                        response = f"The {car_model} has a bore size of {car_info['features']['bore']}."
                    # Check for displacement query
                    elif any(word in user_message.lower() for word in ['displacement', 'engine size', 'capacity']):
                        response = f"The {car_model} has an engine displacement of {car_info['features']['displacement']}."
                    # Check for cylinder count query
                    elif any(word in user_message.lower() for word in ['cylinder', 'cylinders', 'engine']):
                        response = f"The {car_model} has a {car_info['features']['cylinders']}-cylinder engine."
                    # Check for fuel type query
                    elif any(word in user_message.lower() for word in ['fuel', 'gas', 'petrol', 'diesel', 'octane']):
                        response = f"The {car_model} uses {car_info['features']['fuel_type']}."
                    # Generate response based on question type
                    elif question_type == 'performance':
                        response = f"The {car_model} has {car_info['features']['power_PS']} PS ({car_info['features']['power_kW']} kW) of power, accelerates 0-60 mph in {car_info['features']['0-60 mph']}, and has a top speed of {car_info['features']['Top Speed']}."
                    elif question_type == 'pricing':
                        response = f"The {car_model} is priced between {car_info['features']['Price_Range']}."
                    elif question_type == 'customization':
                        response = f"The {car_model} offers {car_info['features']['Customization Options']}. Available colors include {', '.join(car_info['features']['available_colors'])}."
                    else:
                        response = f"The {car_model} is a high-performance sports car with {car_info['features']['power_PS']} PS of power and a top speed of {car_info['features']['Top Speed']}."
                    
                    # Generate suggestions based on the car model
                    suggestions = [
                        f"What's the price of the {car_model}?",
                        f"Tell me about the {car_model}'s performance.",
                        f"What customization options are available for the {car_model}?",
                        f"Compare the {car_model} with other models."
                    ]
                else:
                    # Handle special edition models by providing information about the base model
                    base_model = None
                    if 'turbo' in car_model.lower() and '50' in car_model.lower():
                        base_model = '911 Turbo S'
                    elif 'gt3' in car_model.lower() and 'touring' in car_model.lower():
                        base_model = '911 GT3'
                    elif 'carrera' in car_model.lower() and 'cabriolet' in car_model.lower():
                        base_model = '911 Carrera 4 GTS'
                    elif 'dakar' in car_model.lower():
                        base_model = '911 Carrera 4 GTS'
                    elif 'carrera s' in car_model.lower():
                        base_model = '911 Carrera S'
                    
                    if base_model:
                        car_info = self.get_car_info(base_model)
                        if car_info:
                            if any(word in user_message.lower() for word in ['top speed', 'maximum speed', 'max speed', 'how fast']):
                                response = f"The {base_model} has a top speed of {car_info['features']['Top Speed']}. The {car_model} shares similar performance specifications."
                            elif any(word in user_message.lower() for word in ['power', 'horsepower', 'hp', 'ps', 'kw', 'engine power']):
                                response = f"The {base_model} has {car_info['features']['power_PS']} PS ({car_info['features']['power_kW']} kW) of power. The {car_model} shares similar engine specifications."
                            elif any(word in user_message.lower() for word in ['0-60', 'zero to sixty', 'acceleration', 'how fast']):
                                response = f"The {base_model} accelerates from 0-60 mph in {car_info['features']['0-60 mph']}. The {car_model} shares similar performance specifications."
                            elif any(word in user_message.lower() for word in ['color', 'colour', 'paint', 'exterior']):
                                if 'available_colors' in car_info['features']:
                                    colors = car_info['features']['available_colors']
                                    if isinstance(colors, list):
                                        response = f"The {base_model} is available in the following colors: {', '.join(colors)}. The {car_model} shares similar color options."
                                    else:
                                        response = f"The {base_model} is available in {colors}. The {car_model} shares similar color options."
                                else:
                                    response = f"I don't have specific color information for the {car_model}, but I can tell you about the {base_model} which shares similar specifications. Would you like to know more about the {base_model}?"
                            elif any(word in user_message.lower() for word in ['output per liter', 'power per liter', 'specific output']):
                                response = f"The {base_model} has a maximum output of {car_info['features']['max_output_per_liter_kW']} kW ({car_info['features']['max_output_per_liter_PS']} PS) per liter. The {car_model} shares the same engine specifications."
                            elif any(word in user_message.lower() for word in ['engine speed', 'rpm', 'revolution', 'max engine speed']):
                                response = f"The {base_model} has a maximum engine speed of {car_info['features']['max_engine_speed']} rpm. The {car_model} shares the same engine specifications."
                            elif any(word in user_message.lower() for word in ['stroke', 'piston stroke']):
                                response = f"The {base_model} has a stroke of {car_info['features']['stroke']}. The {car_model} shares the same engine specifications."
                            elif any(word in user_message.lower() for word in ['torque', 'nm', 'newton meters']):
                                response = f"The {base_model} produces {car_info['features']['max_torque']} of torque. The {car_model} shares similar engine specifications."
                            elif any(word in user_message.lower() for word in ['bore', 'bore size', 'cylinder bore']):
                                response = f"The {base_model} has a bore size of {car_info['features']['bore']}. The {car_model} shares the same engine specifications."
                            else:
                                response = f"I don't have specific information about the {car_model}, but I can tell you about the {base_model} which shares similar specifications. Would you like to know more about the {base_model}?"
                            suggestions = [
                                f"What's the price of the {base_model}?",
                                f"Tell me about the {base_model}'s performance.",
                                f"What customization options are available for the {base_model}?",
                                f"Compare the {base_model} with other models."
                            ]
                        else:
                            response = "I'm sorry, I couldn't find information about that specific model. Would you like to know about our current lineup?"
                            suggestions = [
                                "What models do you offer?",
                                "Tell me about your most popular model.",
                                "What's your latest model?",
                                "Show me your sports car lineup."
                            ]
                    else:
                        response = "I'm sorry, I couldn't find information about that specific model. Would you like to know about our current lineup?"
                        suggestions = [
                            "What models do you offer?",
                            "Tell me about your most popular model.",
                            "What's your latest model?",
                            "Show me your sports car lineup."
                        ]
            else:
                # Handle general queries
                for intent in self.intents:
                    if any(pattern.lower() in user_message.lower() for pattern in intent['patterns']):
                        response = random.choice(intent['responses'])
                        suggestions = [
                            "What models do you offer?",
                            "Tell me about your most popular model.",
                            "What's your latest model?",
                            "Show me your sports car lineup."
                        ]
                        break
                
                if not response:
                    response = f"I'm here to help you explore our Porsche lineup. Would you like to know about our models, pricing, or features? Visit 'https://www.porsche.com/usa/models'"
                    suggestions = [
                        "What models do you offer?",
                        "Tell me about your most popular model.",
                        "What's your latest model?",
                        "Show me your sports car lineup."
                    ]
            
            # Enhance response based on sentiment
            response = self.nlp.enhance_response(response, sentiment)
            
            return {
                'response': response,
                'suggestions': suggestions
            }
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return {
                'response': "I apologize, but I encountered an error processing your request. Please try again or ask a different question.",
                'suggestions': []
            }
