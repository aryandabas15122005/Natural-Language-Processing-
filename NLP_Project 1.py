import tkinter as tk
from tkinter import scrolledtext
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

data = {
    'Question': [
        'How much is the admission fee?',
        'How can I apply for a hostel?',
        'When will exams start?',
        'What are the library timings?',
        'How do I pay my semester fees?',
        'Where can I see my timetable?',
        'How can I get my ID card?',
        'Who is the dean of the university?',
        'Is there a placement cell?',
        'What is the refund policy for admission?'
    ],
    'Answer': [
        'Admission fee is ₹5000.',
        'Fill the hostel form online at hostel.university.edu.',
        'Exams will begin in December as per the academic calendar.',
        'Library is open from 9 AM to 8 PM on weekdays.',
        'You can pay your semester fees through the student portal.',
        'Check your timetable on the university website under student section.',
        'Collect your ID card from the admin office after registration.',
        'The dean of the university is Dr. R. Sharma.',
        'Yes, the university has a placement cell to assist students with jobs.',
        'Admission refund is available within 15 days of payment as per policy.'
    ]
}

df = pd.DataFrame(data)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop_words]
    return ' '.join(tokens)

df['Processed'] = df['Question'].apply(preprocess)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Processed'])

def get_response(query):
    q_processed = preprocess(query)
    q_vector = vectorizer.transform([q_processed])
    similarity = cosine_similarity(q_vector, tfidf_matrix)
    idx = similarity.argmax()
    return df.iloc[idx]['Answer']

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
root.title("University FAQ Chatbot")
root.geometry("500x500")
root.resizable(False, False)

chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', font=('Arial', 11))
chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

entry_frame = tk.Frame(root)
entry_frame.pack(pady=10, fill=tk.X)

entry = tk.Entry(entry_frame, font=('Arial', 11))
entry.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.X, expand=True)
entry.bind("<Return>", lambda event: send_message())

send_btn = tk.Button(entry_frame, text="Send", command=send_message, font=('Arial', 11), bg="#0078D7", fg="white")
send_btn.pack(side=tk.RIGHT, padx=10)

root.mainloop()
