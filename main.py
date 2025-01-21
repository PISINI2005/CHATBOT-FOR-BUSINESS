import json
import random
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load intents from JSON file
with open('intents.json') as file:
    data = json.load(file)

intents = data['intents']
corpus = []
labels = []

for intent in intents:
    for example in intent['examples']:
        corpus.append(example)
        labels.append(intent['intent'])

# Train the model using TfidfVectorizer and MultinomialNB
vectorizer = TfidfVectorizer()
classifier = MultinomialNB()
pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', classifier)
])
pipeline.fit(corpus, labels)

def classify_intent(text):
    prediction = pipeline.predict([text])
    return prediction[0]

def get_response(intent):
    responses = {
        "greeting": [
            "Hello! Welcome to our customer service. How can I assist you today?",
            "Hi there! How can I help you with your order or any other issue?",
            "Good day! I'm here to assist you with any questions or concerns.",
            "Greetings! How can I help you today with your purchase or account?",
            "Hey! How can I support you today? Your satisfaction is our priority."
        ],
        "customer_service": [
            "I'm here to help with anything you need. How can I assist you today?",
            "Need help with an order, return, or refund? I'm here to assist!",
            "We value your business. How can I resolve your issue today?",
            "Tell me how I can help. Whether it's tracking an order, processing a return, or anything else.",
            "Customer satisfaction is our top priority. What can I do for you?"
        ],
        "order_tracking": [
            "Please provide your order number, and I'll help you track your package.",
            "Let me assist you with tracking your order. Can you share the order details?",
            "I can help you find out where your order is. Just provide me with the order number.",
            "Need to track your order? I’m here to help—please give me your order information.",
            "I'm here to assist with tracking your package. Just let me know your order number."
        ],
        "returns_refunds": [
            "I can help you with returns and refunds. What seems to be the issue?",
            "Need to process a return or get a refund? I'm here to assist.",
            "Let’s get started with your return or refund. Can you tell me the order details?",
            "I'm here to make your return or refund process easy. Just provide the necessary details.",
            "If you need help with returns or refunds, I’m here to guide you through the process."
        ],
        "general_support": [
            "For any other concerns, I’m here to help. How can I assist?",
            "Whether it's account issues or general inquiries, I’m here to assist you.",
            "I can help with any other questions or concerns. How can I assist you today?",
            "Let me help you with any other issues you may have. How can I support you?",
            "For general support, I’m here to help. What do you need assistance with?"
        ],
        "payment_issues": [
            "I'm sorry to hear about the payment issue. Can you provide more details so I can assist?",
            "Let's look into the payment issue. Please share more information so I can help.",
            "I can help with payment problems. What seems to be the issue?",
            "Let's get your payment issue resolved. Could you provide the transaction details?",
            "I'm here to assist with any billing or payment concerns. How can I help?"
        ],
        "account_management": [
            "I can help you manage your account. What would you like to do?",
            "Need to update your account information? I'm here to assist.",
            "I can guide you through managing your account details. What do you need help with?",
            "Let's update your account. Could you tell me what needs changing?",
            "I'm here to help with your account. What would you like to change or update?"
        ],
        "product_inquiries": [
            "I'd be happy to provide more information about this product. What would you like to know?",
            "Let's talk about this product. What specific details are you looking for?",
            "I can assist with product inquiries. What would you like to learn more about?",
            "Need more details on a product? I'm here to help.",
            "I can help with any product-related questions. What do you need to know?"
        ],
        "technical_support": [
            "I’m sorry you’re experiencing technical issues. Let's troubleshoot this together.",
            "Technical difficulties can be frustrating. I’m here to help resolve this.",
            "Let’s work on fixing the issue you're having. Can you describe the problem?",
            "I’m here to assist with any technical problems you’re facing. What seems to be the issue?",
            "Let’s get your device or service back on track. What technical issue are you dealing with?"
        ],
        "shipping_inquiries": [
            "I can help with shipping questions. What would you like to know?",
            "Need information on shipping? I'm here to assist.",
            "I’m happy to help with your shipping inquiries. What do you need to know?",
            "Let's talk about your shipping options. What can I assist you with?",
            "I can assist with tracking or changing your shipping details. What do you need?"
        ],
        "promotions_discounts": [
            "I can help with any promotions or discounts. What would you like to know?",
            "Need help with a coupon or discount? I'm here to assist.",
            "Let’s make sure you’re getting the best deal. What promo details do you need?",
            "I can assist with applying discounts or finding promotions. How can I help?",
            "Let’s see how we can save you money. What promo or discount questions do you have?"
        ],
        "unknown": [
            "Sorry, I didn’t catch that. Could you please rephrase?",
            "I'm not sure I understood that. Can you clarify your question?",
            "I’m here to help, but I didn’t understand that. Could you explain further?",
            "Could you please rephrase that? I’m not sure I got it.",
            "Sorry, I didn’t get that. Could you please tell me in another way?"
        ]
    }
    return random.choice(responses.get(intent, ["Sorry, I didn't understand that."]))

class ChatbotApp:
    def __init__(self, root):
        self.root = root
        root.title("Business Chatbot")

        # Set window size and background color
        root.geometry("450x550")
        root.configure(bg="#f0f4f5")

        # Load and display the image at the top
        self.image = Image.open(r"C:\Users\DELL\OneDrive\Pictures\Screenshots\Screenshot 2024-10-05 113137.png")
        self.image = self.image.resize((500, 150), Image.Resampling.LANCZOS)  # Resize the image to fit the layout
        self.photo = ImageTk.PhotoImage(self.image)
        self.image_label = tk.Label(root, image=self.photo, bg="#f0f4f5")
        self.image_label.pack(pady=10)

        # Chat history (with custom font and background)
        self.chat_history = scrolledtext.ScrolledText(root, state='disabled', wrap='word', height=20, width=50, bg="#e6f1f3", fg="#333", font=("Arial", 10), bd=0)
        self.chat_history.pack(pady=10)

        # User input field (with custom font and border)
        self.user_input = tk.Entry(root, width=50, font=("Arial", 12), bg="#ffffff", bd=2, relief="solid")
        self.user_input.pack(pady=10)

        # Send button (with custom font, background color, and hover effect)
        self.send_button = tk.Button(root, text="Send", command=self.send_message, bg="#4caf50", fg="white", font=("Arial", 12, "bold"), bd=0, padx=10, pady=5, activebackground="#45a049")
        self.send_button.pack(pady=5)

    def send_message(self):
        user_text = self.user_input.get()
        if user_text:
            self.chat_history.config(state='normal')
            self.chat_history.insert(tk.END, f"You: {user_text}\n")
            intent = classify_intent(user_text)
            response = get_response(intent)
            self.chat_history.insert(tk.END, f"Bot: {response}\n")
            self.chat_history.config(state='disabled')
            self.chat_history.yview(tk.END) 
            self.user_input.delete(0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()
