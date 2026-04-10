# Receives URL from Vue and sends results back to the frontend in JSON form
from flask import Flask, request, jsonify
from flask_cors import CORS
# Finds the article text inside the HTML
from bs4 import BeautifulSoup
# Built-in Python tool used to break apart a URL
from urllib.parse import urlparse
# Analyzes each sentence for bias and adjectives
from textblob import TextBlob
# Searches the internet for the HTML of the entered URL
import requests
# Breaks the article into individual sentences
import nltk

app = Flask(__name__)
CORS(app)

# Score Label
def get_subjectivity_label(score):
    if score <= 25: return "Neutral"
    if score < 60: return "Balanced"
    if score < 75: return "Opinionated"
    return "Highly Subjective"

def analyze_page_content(url):
    try:
        # Gets and cleans the URL
        publisher_name = urlparse(url).netloc.replace('www.', '')
        # Mimics a real browser to prevent being blocked by news sites
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract the title and text
        page_title = soup.title.string if soup.title else "Unknown Title"
        # Gets text from <p> tags to avoid other unnecessary text
        paragraphs = soup.find_all('p')
        # Skips common footer/header noise
        ignore_list = ["all rights reserved", "©", "copyright", "google play", "apple store", "terms of service"]
        cleaned_paragraphs = []
        for p in paragraphs:
            text = p.get_text().strip()
            # Only add if it's not in our ignore list and has a decent length
            if not any(word in text.lower() for word in ignore_list) and len(text) > 30:
                cleaned_paragraphs.append(text)
        # Split the text into sentences
        sentences = nltk.sent_tokenize(" ".join(cleaned_paragraphs))
        detailed_analysis = []
        total_risk_sum = 0  
        # Analyze each sentence
        for sentence in sentences:
            clean_sentence = sentence.strip()
            # Skip tiny sentence fragments
            if len(clean_sentence) < 20: continue

            blob = TextBlob(clean_sentence)            
            # Subjectivity Bias (0.0 is a Fact and 1.0 is an Opinion)
            subj_score = blob.sentiment.subjectivity * 100
            # Adjective Density (News with too many adjectives is usually sensationalist)
            adj_count = len([tag for word, tag in blob.tags if tag == 'JJ'])
            adj_ratio = (adj_count / len(blob.words)) * 100 if len(blob.words) > 0 else 0
            # Combine for sentence risk score
            sentence_risk = (subj_score + adj_ratio) / 2
            # Subjectivity score
            subj_label = get_subjectivity_label(sentence_risk)
            # Tooltip data
            explanation = (
                f"Verdict: {subj_label} ({round(sentence_risk)}%) | "
                f"Subjectivity: {round(subj_score)}% | "
                f"Adjective Density: {round(adj_ratio)}%"
            )
            detailed_analysis.append({
                "text": clean_sentence,
                "risk_score": round(sentence_risk),
                "details": explanation
            })
            total_risk_sum += sentence_risk

        # Calculate overall credibility
        avg_risk = (total_risk_sum / len(detailed_analysis)) if detailed_analysis else 0
        risk_score = round(avg_risk) 
        # Determine the verdict label
        if risk_score < 25:
            verdict = "Low Risk: Content appears neutral."
        elif risk_score < 75:
            verdict = "Moderate Risk: Significant biased language found."
        else:
            verdict = "High Risk: Sensationalist or biased patterns found."
        # Return the data
        return {
            "title": page_title,
            "publisher": publisher_name,
            "risk_gauge": risk_score,
            "verdict": verdict,
            "sentences": detailed_analysis
        }

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

@app.route('/analyse', methods=['POST'])
def analyse_url():
    data = request.json
    url = data.get('url')
    print(f"Scraping & Analyzing: {url}")
    result = analyze_page_content(url)
    # Send the result to the frontend
    return jsonify({
        "status": "success",
        "message": result,
        "received_url": url
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)