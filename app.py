import os
import base64
import json
import re
from flask import Flask, request, jsonify, render_template
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = Anthropic()

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp"}
MEDIA_TYPE_MAP = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}

OCR_SYSTEM_PROMPT = (
    "You are an expert OCR engine. Your only job is to transcribe handwritten or printed "
    "text from images exactly as it appears, preserving paragraph breaks with blank lines. "
    "Do not correct spelling, grammar, or punctuation. Do not add any commentary. "
    "Output only the transcribed text."
)

MARKING_SYSTEM_PROMPT = """You are an experienced PSLE (Primary School Leaving Examination) English Paper 1 Continuous Writing marker with 15 years of marking experience. You assess compositions strictly according to MOE Singapore's PSLE marking rubric.

SCORING RUBRIC (36 marks total):

CONTENT (18 marks) — evaluate:
- Relevance: Does the story address the given topic/prompt?
- Idea development: Are ideas elaborated with details and depth?
- Plot coherence: Is there a logical sequence of events?
- Story arc: Does the composition have a clear Beginning → Rising Action → Conflict/Climax → Resolution?
- Character and setting: Are they established clearly?

LANGUAGE (18 marks) — evaluate:
- Grammar and syntax: Subject-verb agreement, tense consistency, sentence structure
- Vocabulary: Word choice precision, variety, appropriateness for P6 level
- Sentence variety: Mix of simple, compound, and complex sentences
- Paragraph organisation: Clear topic sentences, smooth transitions between paragraphs
- Tense consistency: Does the student maintain a consistent tense throughout?

ISSUE CATEGORIES:
- "logic": plot holes, contradictions, unrealistic events, timeline inconsistencies
- "flow": abrupt transitions, poor paragraphing, disjointed pacing, missing connectives
- "language": grammar errors, wrong tenses, awkward vocabulary, run-on sentences, spelling
- "content": off-topic sections, underdeveloped ideas, missing story arc elements

SEVERITY LEVELS:
- "high": Significantly detracts from the quality; would cause substantial mark deduction
- "medium": Noticeable issue that moderately affects the writing
- "low": Minor issue; style or optional improvement

You MUST respond with valid JSON only. No prose before or after the JSON object."""


def extract_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ocr", methods=["POST"])
def ocr():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image file provided"}), 400

        file = request.files["image"]
        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"success": False, "error": f"Unsupported file type: {ext}"}), 400

        media_type = MEDIA_TYPE_MAP[ext]
        image_data = base64.standard_b64encode(file.read()).decode("utf-8")

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=OCR_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Transcribe all text visible in this image. Preserve paragraph structure using blank lines between paragraphs. Output only the transcribed text.",
                        },
                    ],
                }
            ],
        )

        return jsonify({"success": True, "text": response.content[0].text})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/mark", methods=["POST"])
def mark():
    try:
        data = request.get_json()
        if not data or not data.get("essay", "").strip():
            return jsonify({"success": False, "error": "Essay text is required"}), 400

        essay = data["essay"].strip()

        user_message = f"""Mark the following PSLE English Continuous Writing composition.

ESSAY:
\"\"\"
{essay}
\"\"\"

Respond with a JSON object in this EXACT schema — no other text:
{{
  "scores": {{
    "content": <integer 0-18>,
    "language": <integer 0-18>,
    "total": <integer 0-36>
  }},
  "issues": [
    {{
      "snippet": "<exact short phrase from the essay, 3-15 words>",
      "type": "<one of: logic | flow | language | content>",
      "explanation": "<1-2 sentences explaining the problem>",
      "suggestion": "<concrete corrected text or advice>",
      "severity": "<one of: high | medium | low>"
    }}
  ],
  "strengths": ["<strength 1>", "<strength 2>"],
  "improvements": ["<key improvement 1>", "<key improvement 2>"],
  "overall_comment": "<2-3 sentence holistic comment a teacher would write>"
}}

Rules:
- "snippet" must be a verbatim excerpt from the essay (used for text highlighting)
- "total" must equal "content" + "language"
- Include 3-8 issues; do not flag trivial issues with severity "low" unless the essay is otherwise strong
- "strengths" should have 2-4 items; "improvements" should have 2-4 items
- If the essay is very short (under 100 words), set content score <= 8 and note underdevelopment"""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=MARKING_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        result = extract_json(response.content[0].text)
        return jsonify({"success": True, "result": result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
