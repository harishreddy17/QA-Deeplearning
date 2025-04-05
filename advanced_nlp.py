import spacy
from textblob import TextBlob
import json
import re
import random

class AdvancedNLP:
    def __init__(self):
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            
            # Load car data for better entity recognition
            with open('cars.json', 'r') as f:
                self.car_data = json.load(f)
            
            # Extract car models for matching - store both full names and parts
            self.car_models = []
            self.car_model_parts = {}
            for model in self.car_data['models']:
                model_name = model['name']
                self.car_models.append(model_name)
                # Store parts for each model
                self.car_model_parts[model_name] = {
                    'full_name': model_name,
                    'parts': model_name.lower().split(),
                    'length': len(model_name.split())
                }
            
        except Exception as e:
            print(f"Error initializing AdvancedNLP: {str(e)}")
            raise

    def extract_entities(self, text):
        """
        Extract entities from the text using spaCy and custom car model matching.
        """
        try:
            # Convert text to lowercase for consistent comparison
            text_lower = text.lower()
            
            # First try exact match with car models
            for model in self.car_models:
                if text_lower == model.lower():
                    return {'car_model': model}
            
            # If no exact match, try partial match with priority to longer names
            matching_models = []
            for model in self.car_models:
                model_parts = model.lower().split()
                text_parts = text_lower.split()
                
                # Check if all parts of the model name appear in the text in order
                is_match = True
                current_pos = 0
                matched_parts = 0
                
                for part in model_parts:
                    found = False
                    for i in range(current_pos, len(text_parts)):
                        if text_parts[i] == part:
                            current_pos = i + 1
                            matched_parts += 1
                            found = True
                            break
                    if not found:
                        is_match = False
                        break
                
                # Only consider it a match if we found all parts in order
                if is_match and matched_parts == len(model_parts):
                    matching_models.append((model, len(model_parts)))
            
            if matching_models:
                # Sort by length of model name (longer = more specific)
                matching_models.sort(key=lambda x: -x[1])
                return {'car_model': matching_models[0][0]}
            
            # If no car model found, use spaCy NER for other entities
            doc = self.nlp(text)
            entities = {}
            for ent in doc.ents:
                if ent.label_ not in entities:
                    entities[ent.label_] = ent.text
            
            return entities
            
        except Exception as e:
            print(f"Error extracting entities: {str(e)}")
            return {}

    def analyze_sentiment(self, text):
        try:
            analysis = TextBlob(text)
            return analysis.sentiment.polarity
        except Exception as e:
            print(f"Error analyzing sentiment: {str(e)}")
            return 0.0

    def classify_question(self, text):
        try:
            text = text.lower()
            
            # Performance-related questions
            if any(word in text for word in ['speed', 'power', 'horsepower', 'acceleration', '0-60', 'top speed', 'performance', 'engine']):
                return 'performance'
            
            # Pricing-related questions
            elif any(word in text for word in ['price', 'cost', 'how much', 'expensive', 'afford']):
                return 'pricing'
            
            # Customization-related questions
            elif any(word in text for word in ['custom', 'color', 'option', 'package', 'feature', 'interior', 'exterior']):
                return 'customization'
            
            # Feature-related questions
            elif any(word in text for word in ['feature', 'spec', 'specification', 'detail', 'include', 'have']):
                return 'features'
            
            # Default to general
            return 'general'
            
        except Exception as e:
            print(f"Error classifying question: {str(e)}")
            return 'general'

    def enhance_response(self, response, sentiment):
        try:
            # Add enthusiasm for positive sentiment
            if sentiment > 0.3:
                enhancements = [
                    "Absolutely! ",
                    "Great choice! ",
                    "Excellent question! ",
                    "I'm excited to tell you that "
                ]
                response = random.choice(enhancements) + response
            
            # Add reassurance for negative sentiment
            elif sentiment < -0.3:
                enhancements = [
                    "I understand your concern. ",
                    "Let me clarify that ",
                    "I want to assure you that ",
                    "To address your question, "
                ]
                response = random.choice(enhancements) + response
            
            # Add follow-up for neutral sentiment
            else:
                if not response.endswith('?'):
                    response += " Would you like to know more?"
            
            return response
            
        except Exception as e:
            print(f"Error enhancing response: {str(e)}")
            return response 