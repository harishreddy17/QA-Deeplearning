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
            # Clean up the input car name
            car_name = car_name.lower().strip()
            
            # First try exact match (case-insensitive)
            for model in self.car_data['models']:
                if car_name == model['name'].lower():
                    return model
            
            # If no exact match, try to find the most specific match
            best_match = None
            best_score = 0
            
            for model in self.car_data['models']:
                model_name = model['name'].lower()
                score = 0
                
                # Check if all words in the model name are present in the input
                model_words = set(model_name.split())
                input_words = set(car_name.split())
                
                # Count matching words
                matching_words = model_words.intersection(input_words)
                if len(matching_words) > 0:
                    # Base score on number of matching words
                    score = len(matching_words)
                    
                    # Bonus for exact model number match (e.g., "911")
                    if any(word.isdigit() for word in matching_words):
                        score += 2
                    
                    # Bonus for matching the full model name
                    if model_name in car_name:
                        score += 3
                    
                    # Special handling for GTS models
                    if 'gts' in car_name and 'gts' in model_name:
                        score += 2
                        # Additional bonus for Carrera GTS
                        if 'carrera' in car_name and 'carrera' in model_name:
                            score += 2
                    
                    # Special handling for Carrera models
                    if 'carrera' in car_name and 'carrera' in model_name:
                        score += 1
                        # Additional bonus for Carrera T
                        if 't' in car_name and 't' in model_name:
                            score += 2
                    
                    # Special handling for Turbo models
                    if 'turbo' in car_name and 'turbo' in model_name:
                        score += 2
                    
                    # Special handling for GT3 models
                    if 'gt3' in car_name and 'gt3' in model_name:
                        score += 2
                    
                    # Update best match if this is better
                    if score > best_score:
                        best_score = score
                        best_match = model
            
            # Only return a match if we have a good confidence score
            if best_score >= 2:
                return best_match
            
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

    def get_response(self, user_message, state=None):
        try:
            # Initialize response components
            response = ""
            suggestions = []
            
            # Handle conversation state
            if state is None:
                state = {
                    'isFirstInteraction': True,
                    'hasProvidedName': False,
                    'hasShownMainMenu': False,
                    'selectedModel': None,
                    'userName': None
                }
            
            # First interaction - ask for name
            if state['isFirstInteraction']:
                state['isFirstInteraction'] = False
                response = "Welcome to Porsche! I am your Virtual Assistant. May I know your name?"
                suggestions = []
                return {
                    'response': response,
                    'suggestions': suggestions,
                    'state': state
                }
            
            # If we have the user's name but haven't shown the main menu yet
            if not state['hasProvidedName']:
                state['hasProvidedName'] = True
                state['userName'] = user_message.strip()
                state['hasShownMainMenu'] = True
                response = f"Hi {state['userName']}! How can we help you today?"
                suggestions = [
                    "Explore our models",
                    "Find a dealership",
                    "Schedule a test drive",
                    "Learn about customization options"
                ]
                return {
                    'response': response,
                    'suggestions': suggestions,
                    'state': state
                }
            
            # Check if the message contains a model name and feature query
            car_info = self.get_car_info(user_message)
            if car_info:
                # Handle specific feature queries
                if 'engine' in user_message.lower() or 'motor' in user_message.lower():
                    response = f"""The {car_info['name']} features a powerful {car_info['features']['cylinders']}-cylinder engine:
• Displacement: {car_info['features']['displacement']}
• Bore: {car_info['features']['bore']}
• Stroke: {car_info['features']['stroke']}
• Max Engine Speed: {car_info['features']['max_engine_speed']} rpm"""
                elif 'transmission' in user_message.lower() or 'gearbox' in user_message.lower():
                    response = f"The {car_info['name']} comes with a {car_info['features']['transmission']} transmission system."
                elif 'fuel' in user_message.lower() or 'economy' in user_message.lower() or 'mileage' in user_message.lower():
                    response = f"""The {car_info['name']} has the following fuel specifications:
• Fuel Type: {car_info['features']['fuel_type']}
• Fuel Economy: {car_info['features']['fuel_economy']}"""
                elif 'warranty' in user_message.lower() or 'guarantee' in user_message.lower():
                    response = f"The {car_info['name']} comes with a comprehensive {car_info['features']['warranty']} warranty."
                elif 'dimensions' in user_message.lower() or 'size' in user_message.lower():
                    response = f"""The {car_info['name']} has the following dimensions:
• Length: {car_info['features']['length']}
• Width: {car_info['features']['width']}
• Height: {car_info['features']['height']}
• Wheelbase: {car_info['features']['wheelbase']}"""
                elif 'weight' in user_message.lower() or 'mass' in user_message.lower():
                    response = f"The {car_info['name']} has a curb weight of {car_info['features']['curb_weight']}."
                elif 'seats' in user_message.lower() or 'seating' in user_message.lower():
                    response = f"The {car_info['name']} has a seating capacity of {car_info['features']['seating_capacity']}."
                elif 'trunk' in user_message.lower() or 'boot' in user_message.lower() or 'storage' in user_message.lower():
                    response = f"The {car_info['name']} offers {car_info['features']['trunk_volume']} of trunk space."
                elif 'customization' in user_message.lower() or 'colors' in user_message.lower() or 'options' in user_message.lower():
                    response = f"""The {car_info['name']} offers {car_info['features']['Customization Options']}. 
Available colors include {', '.join(car_info['features']['available_colors'])}.
You can explore these options and more at <a href='https://www.porsche.com' target='_blank'>porsche.com</a>"""
                elif 'performance' in user_message.lower() or 'speed' in user_message.lower() or 'power' in user_message.lower():
                    response = f"""The {car_info['name']} offers exceptional performance:
• Power: {car_info['features']['power_PS']} PS ({car_info['features']['power_kW']} kW)
• Acceleration: 0-60 mph in {car_info['features']['0-60 mph']}
• Top Speed: {car_info['features']['Top Speed']}
• Max Torque: {car_info['features']['max_torque']}
• Engine: {car_info['features']['cylinders']}-cylinder, {car_info['features']['displacement']}"""
                elif 'price' in user_message.lower() or 'cost' in user_message.lower():
                    response = f"The {car_info['name']} is priced between {car_info['features']['Price_Range']}."
                else:
                    # Default comprehensive feature overview
                    response = f"""The {car_info['name']} is a high-performance sports car with the following features:
• Engine: {car_info['features']['cylinders']}-cylinder, {car_info['features']['displacement']}
• Power: {car_info['features']['power_PS']} PS ({car_info['features']['power_kW']} kW)
• Acceleration: 0-60 mph in {car_info['features']['0-60 mph']}
• Top Speed: {car_info['features']['Top Speed']}
• Transmission: {car_info['features']['transmission']}
• Seating: {car_info['features']['seating_capacity']}
• Trunk Space: {car_info['features']['trunk_volume']}
• Available Colors: {', '.join(car_info['features']['available_colors'])}"""
                
                # Generate relevant suggestions based on the query
                suggestions = [
                    f"What's the price of the {car_info['name']}?",
                    f"Tell me about the {car_info['name']}'s performance",
                    f"What customization options are available for the {car_info['name']}?",
                    f"What's the {car_info['name']}'s fuel economy?",
                    f"What are the {car_info['name']}'s dimensions?",
                    f"What's the {car_info['name']}'s warranty coverage?",
                    "Back to models list",
                    "Back to main menu"
                ]
                return {
                    'response': response,
                    'suggestions': suggestions,
                    'state': state
                }
            
            # Handle main menu selection
            if state['hasShownMainMenu'] and not state['selectedModel']:
                if 'models' in user_message.lower():
                    # Get all model names
                    model_names = [model['name'] for model in self.car_data['models']]
                    response = "Here are our available models. Please select one to learn more:"
                    suggestions = model_names + ["Back to main menu"]
                    return {
                        'response': response,
                        'suggestions': suggestions,
                        'state': state
                    }
                elif 'dealership' in user_message.lower():
                    response = "You can find your nearest Porsche dealership at <a href='https://www.porsche.com' target='_blank'>porsche.com</a>"
                    suggestions = ["Back to main menu", "Schedule a test drive"]
                    return {
                        'response': response,
                        'suggestions': suggestions,
                        'state': state
                    }
                elif 'test drive' in user_message.lower():
                    response = "You can schedule a test drive at <a href='https://www.porsche.com' target='_blank'>porsche.com</a>"
                    suggestions = ["Back to main menu", "Find a dealership"]
                    return {
                        'response': response,
                        'suggestions': suggestions,
                        'state': state
                    }
                elif 'customization' in user_message.lower():
                    response = "You can explore our customization options at <a href='https://www.porsche.com' target='_blank'>porsche.com</a>"
                    suggestions = ["Back to main menu", "Explore our models"]
                    return {
                        'response': response,
                        'suggestions': suggestions,
                        'state': state
                    }
                elif 'back to main menu' in user_message.lower():
                    response = f"How can we help you today, {state['userName']}?"
                    suggestions = [
                        "Explore our models",
                        "Find a dealership",
                        "Schedule a test drive",
                        "Learn about customization options"
                    ]
                    return {
                        'response': response,
                        'suggestions': suggestions,
                        'state': state
                    }
            
            # Handle model selection
            if state['hasShownMainMenu']:
                # Check if the user selected a model
                car_info = self.get_car_info(user_message)
                if car_info:
                    state['selectedModel'] = car_info['name']
                    response = f"You've selected the {state['selectedModel']}. What would you like to know about it?"
                    suggestions = [
                        f"What's the price of the {state['selectedModel']}?",
                        f"Tell me about the {state['selectedModel']}'s performance",
                        f"What customization options are available for the {state['selectedModel']}?",
                        "Back to models list",
                        "Back to main menu"
                    ]
                    return {
                        'response': response,
                        'suggestions': suggestions,
                        'state': state
                    }
                
                # Handle back to models list
                if 'back to models list' in user_message.lower():
                    state['selectedModel'] = None
                    response = "Here are our available models. Please select one to learn more:"
                    suggestions = [model['name'] for model in self.car_data['models']] + ["Back to main menu"]
                    return {
                        'response': response,
                        'suggestions': suggestions,
                        'state': state
                    }
            
            # Handle specific model queries
            if state['selectedModel']:
                car_info = self.get_car_info(state['selectedModel'])
                if car_info:
                    if 'back to main menu' in user_message.lower():
                        state['selectedModel'] = None
                        response = f"How can we help you today, {state['userName']}?"
                        suggestions = [
                            "Explore our models",
                            "Find a dealership",
                            "Schedule a test drive",
                            "Learn about customization options"
                        ]
                        return {
                            'response': response,
                            'suggestions': suggestions,
                            'state': state
                        }
                    elif 'price' in user_message.lower() or 'cost' in user_message.lower():
                        response = f"The {state['selectedModel']} is priced between {car_info['features']['Price_Range']}."
                    elif 'performance' in user_message.lower():
                        response = f"""The {state['selectedModel']} offers exceptional performance:
• Power: {car_info['features']['power_PS']} PS ({car_info['features']['power_kW']} kW)
• Acceleration: 0-60 mph in {car_info['features']['0-60 mph']}
• Top Speed: {car_info['features']['Top Speed']}"""
                    elif 'customization' in user_message.lower():
                        response = f"The {state['selectedModel']} offers {car_info['features']['Customization Options']}. Available colors include {', '.join(car_info['features']['available_colors'])}."
                    else:
                        response = f"The {state['selectedModel']} is a high-performance sports car with {car_info['features']['power_PS']} PS of power and a top speed of {car_info['features']['Top Speed']}."
                    
                    suggestions = [
                        f"What's the price of the {state['selectedModel']}?",
                        f"Tell me about the {state['selectedModel']}'s performance",
                        f"What customization options are available for the {state['selectedModel']}?",
                        "Back to models list",
                        "Back to main menu"
                    ]
                    return {
                        'response': response,
                        'suggestions': suggestions,
                        'state': state
                    }
            
            # Default response
            response = f"I'm here to help you explore our Porsche lineup. Would you like to know about our models, pricing, or features? Visit <a href='https://www.porsche.com' target='_blank'>porsche.com</a>"
            suggestions = [
                "What models do you offer?",
                "Tell me about your most popular model.",
                "What's your latest model?",
                "Show me your sports car lineup."
            ]
            
            return {
                'response': response,
                'suggestions': suggestions,
                'state': state
            }
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return {
                'response': "I apologize, but I encountered an error processing your request. Please try again or ask a different question.",
                'suggestions': [],
                'state': state
            }
