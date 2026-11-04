import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from openai import APIConnectionError, APIStatusError, OpenAI

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")

# Groq exposes an OpenAI-compatible API, so the openai package works as is.
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)
MODEL = "openai/gpt-oss-120b"


def ask_model(system_prompt, user_prompt, temperature, max_tokens=512):
    """Send one chat request and return the reply text."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def respond(system_prompt, user_prompt, temperature):
    """Call the model and always return JSON, even when something fails."""
    try:
        answer = ask_model(system_prompt, user_prompt, temperature)
        return jsonify({"response": answer}), 200
    except APIStatusError as e:
        print(f"API error {e.status_code}: {e.response.text}")
        if e.status_code == 401:
            message = "Invalid API key. Check GROQ_API_KEY in your .env file."
        elif e.status_code == 429:
            message = "Rate limit reached. Please wait a moment and try again."
        else:
            message = f"The API returned an error ({e.status_code}). Check the terminal for details."
        return jsonify({"response": message}), e.status_code
    except APIConnectionError:
        return jsonify({"response": "Could not reach the API. Check your internet connection."}), 503
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"response": "Something went wrong on the server."}), 500


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    question = (request.form.get("question") or "").strip()
    if not question:
        return jsonify({"response": "Please enter a question."}), 400

    return respond("Act like a helpful personal assistant", question, temperature=0.7)


@app.route("/summarize", methods=["POST"])
def summarize():
    email_text = (request.form.get("email") or "").strip()
    if not email_text:
        return jsonify({"response": "Please paste an email to summarize."}), 400

    prompt = f"Summarize the following email in 2-3 sentences:\n\n{email_text}"
    return respond("Act like an expert email assistant", prompt, temperature=0.3)


if __name__ == "__main__":
    app.run(debug=True)