import tkinter as tk
from tkinter import scrolledtext
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

data = {
    'Intent': [
        'Order Status',
        'Return Policy',
        'Product Info',
        'Delivery Time',
        'Cancel Order',
        'Payment Issue',
        'Exchange Policy',
        'Warranty Info',
        'Shipping Charges',
        'Contact Support'
    ],
    'Example Query': [
        'Where is my order #12345?',
        'How can I return a product?',
        'Does this phone support fast charging?',
        'When will my package arrive?',
        'Can I cancel my order?',
        'My payment failed but money was deducted.',
        'Can I exchange this item?',
        'What is the warranty period for this laptop?',
        'How much are the shipping charges?',
        'How do I contact customer support?'
    ],
    'Response': [
        'Your order {order_no} is out for delivery.',
        'You can return products within 15 days via our online portal.',
        'Yes, this phone supports fast charging.',
        'Most orders are delivered within 3-5 business days.',
        'You can cancel your order within 24 hours from the order page.',
        'If payment failed but money was deducted, it will be refunded in 3-5 days.',
        'You can exchange products within 7 days if they are unused and in original packaging.',
        'This laptop comes with a 1-year manufacturer warranty.',
        'Shipping charges are ₹50 for orders below ₹500. Orders above ₹500 are free.',
        'You can reach customer support at support@shoponline.com or call 1800-555-123.'
    ]
}

df = pd.DataFrame(data)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop_words]
    return ' '.join(tokens)

df['Processed'] = df['Example Query'].apply(preprocess)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Processed'])

def extract_order_number(text):
    match = re.search(r'#\d+', text)
    return match.group() if match else None

def get_response(query):
    q_processed = preprocess(query)
    q_vector = vectorizer.transform([q_processed])
    similarity = cosine_similarity(q_vector, tfidf_matrix)
    idx = similarity.argmax()
    response = df.iloc[idx]['Response']
    order_no = extract_order_number(query)
    if order_no:
        response = response.format(order_no=order_no)
    else:
        response = response.format(order_no='N/A')
    return response

def send_message():
    user_msg = entry.get()
    if user_msg.strip() == '':
        return
    chat_window.config(state='normal')
    chat_window.insert(tk.END, "You: " + user_msg + "\n")
    response = get_response(user_msg)
    chat_window.insert(tk.END, "Chatbot: " + response + "\n\n")
    chat_window.config(state='disabled')
    chat_window.yview(tk.END)
    entry.delete(0, tk.END)

root = tk.Tk()
root.title("Customer Support Chatbot")
root.geometry("520x520")
root.resizable(False, False)

chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', font=('Arial', 11))
chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

entry_frame = tk.Frame(root)
entry_frame.pack(pady=10, fill=tk.X)

entry = tk.Entry(entry_frame, font=('Arial', 11))
entry.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.X, expand=True)
entry.bind("<Return>", lambda event: send_message())

send_btn = tk.Button(entry_frame, text="Send", command=send_message, font=('Arial', 11), bg="#28a745", fg="white")
send_btn.pack(side=tk.RIGHT, padx=10)

root.mainloop()
