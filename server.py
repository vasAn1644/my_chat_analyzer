from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
import json
import groq
import os
from datetime import datetime
from dotenv import load_dotenv

# Завантажуємо змінні з .env
load_dotenv()

# Ініціалізуємо клієнт Groq API
client = groq.Client(api_key=os.getenv('API_KEY'))

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Максимальна довжина одного запиту до API (6000 символів)
MAX_CHARS = 6000
# Максимальний розмір файлу для повного читання (2500 КБ = 2.5 МБ)
MAX_FILE_SIZE_KB = 2500
# Максимальна кількість повідомлень для обробки
MAX_MESSAGES = 2500

def split_text(text, max_length=MAX_CHARS):
    """Розбиває текст на частини, які не перевищують max_length символів."""
    if len(text) <= max_length:
        return [text]
    
    parts = []
    current_part = ""
    
    for line in text.split("\n"):
        if len(current_part) + len(line) + 1 <= max_length:
            current_part += line + "\n"
        else:
            if current_part:
                parts.append(current_part.strip())
            current_part = line + "\n"
    
    if current_part:
        parts.append(current_part.strip())
    
    return parts

def read_json_with_limit(file_path, max_size_kb=MAX_FILE_SIZE_KB, max_messages=MAX_MESSAGES):
    """Читає JSON і повертає останні max_messages повідомлень, якщо файл великий."""
    file_size = os.path.getsize(file_path) / 1024  # Розмір у КБ
    print(f"Розмір файлу: {file_size:.2f} КБ")
    
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        messages = data.get("messages", [])
        
        if file_size > max_size_kb:
            print(f"Файл великий, обробляємо лише останні {max_messages} повідомлень")
            return {"messages": messages[-max_messages:]}
        return data

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "Файл не знайдено"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "Файл не вибрано"}), 400

    if file and file.filename.endswith('.json'):
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], "result.json")
        file.save(file_path)
        return redirect(url_for('index'))

    return jsonify({"error": "Невірний формат файлу. Дозволені тільки .json"}), 400

@app.route('/analysis')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], "result.json")
        if not os.path.exists(file_path):
            return jsonify({"error": "Файл чату не знайдено"}), 400
        
        data = read_json_with_limit(file_path)
        messages = data.get("messages", [])
        if not messages:
            return jsonify({"error": "Файл не містить повідомлень"}), 400

        user_question = request.json.get("question", "").strip()
        if not user_question:
            return jsonify({"error": "Питання не може бути порожнім"}), 400
        
        chat_context = "\n".join(
            [f"{msg.get('from', 'Невідомий')}: {msg.get('text', '')}" for msg in messages if isinstance(msg.get('text'), str)]
        )[:2000]

        full_prompt = f"Чат:\n{chat_context}\n\nПитання: {user_question}"
        prompt_parts = split_text(full_prompt, MAX_CHARS)

        responses = []
        for part in prompt_parts:
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "Ти аналізатор чату. Відповідай, базуючись на історії переписки."},
                    {"role": "user", "content": part}
                ]
            )
            answer = response.choices[0].message.content.strip() if response.choices else "Не вдалося знайти відповідь."
            responses.append(answer)

        final_answer = " ".join(responses)
        return jsonify({"answer": final_answer})

    except Exception as e:
        print(f"Помилка у /ask: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], "result.json")
        print(f"Перевірка файлу: {file_path}")
        if not os.path.exists(file_path):
            return jsonify({"error": "Файл result.json не знайдено"}), 400

        data = read_json_with_limit(file_path)
        messages = data.get("messages", [])
        print(f"Кількість повідомлень: {len(messages)}")

        participants = {}
        words_dict = {}
        total_date_dict = {}
        weekly_activity = {str(i): 0 for i in range(7)}
        hourly_activity = {str(i): 0 for i in range(24)}
        min_word_length = 3

        for msg in messages:
            if msg["type"] == "message":
                sender = msg.get("forwarded_from", "Невідомий") if "forwarded_from" in msg else msg.get("from", "Невідомий")
                participants[sender] = participants.get(sender, 0) + 1
                
                date_str = msg['date']
                date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
                total_date_dict[date_str[:10]] = total_date_dict.get(date_str[:10], 0) + 1
                weekly_activity[str(date.weekday())] += 1
                hourly_activity[str(date.hour)] += 1
                
                text = msg.get('text', '')
                if isinstance(text, str):
                    for word in text.lower().split():
                        if len(word) > min_word_length:
                            words_dict[word] = words_dict.get(word, 0) + 1
                elif isinstance(text, list):
                    combined_text = ''.join(
                        item['text'] if isinstance(item, dict) and 'text' in item else str(item)
                        for item in text
                    )
                    for word in combined_text.lower().split():
                        if len(word) > min_word_length:
                            words_dict[word] = words_dict.get(word, 0) + 1

        most_used_words = sorted(words_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        most_active_dates = sorted(total_date_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        average_messages_per_day = sum(total_date_dict.values()) / len(total_date_dict) if total_date_dict else 0

        return jsonify({
            "total_messages": sum(participants.values()),
            "most_used_words": most_used_words,
            "average_messages_per_day": round(average_messages_per_day, 2),
            "most_active_dates": most_active_dates,
            "weekly_activity": weekly_activity,
            "hourly_activity": hourly_activity,
            "participants": participants,
            "is_truncated": os.path.getsize(file_path) / 1024 > MAX_FILE_SIZE_KB
        })
    except Exception as e:
        print(f"Помилка у /stats: {str(e)}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)