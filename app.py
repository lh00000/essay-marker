import os
import base64
import json
import re
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    + GEMINI_MODEL
    + ":generateContent"
)

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

CONTENT (18 marks) - evaluate:
- Relevance: Does the story address the given topic/prompt?
- Idea development: Are ideas elaborated with details and depth?
- Plot coherence: Is there a logical sequence of events?
- Story arc: Does the composition have a clear Beginning, Rising Action, Conflict/Climax, Resolution?
- Character and setting: Are they established clearly?

LANGUAGE (18 marks) - evaluate:
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


def gemini_request(system, parts, max_tokens=4096):
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"maxOutputTokens": max_tokens},
    }
    resp = requests.post(
        GEMINI_URL,
        params={"key": api_key},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def extract_json(text):
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except ValueError:
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
            return jsonify({"success": False, "error": "Unsupported file type: " + ext}), 400

        media_type = MEDIA_TYPE_MAP[ext]
        image_data = base64.b64encode(file.read()).decode("utf-8")

        parts = [
            {"inline_data": {"mime_type": media_type, "data": image_data}},
            {
                "text": (
                    "Transcribe all text visible in this image. "
                    "Preserve paragraph structure using blank lines between paragraphs. "
                    "Output only the transcribed text."
                )
            },
        ]

        text = gemini_request(system=OCR_SYSTEM_PROMPT, parts=parts)
        return jsonify({"success": True, "text": text})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/mark", methods=["POST"])
def mark():
    try:
        data = request.get_json()
        if not data or not data.get("essay", "").strip():
            return jsonify({"success": False, "error": "Essay text is required"}), 400

        essay = data["essay"].strip()
        question = data.get("question", "").strip()

        question_block = ""
        question_schema = ""
        question_rules = ""
        if question:
            question_block = (
                "ESSAY QUESTION / THEME:\n"
                '"""\n' + question + '\n"""\n\n'
                "Before marking, dissect this question to identify:\n"
                "1. The core theme or topic the student must address\n"
                "2. The specific requirements or expectations implied by the question\n"
                "3. How well the submitted essay actually addresses the question\n\n"
            )
            question_schema = (
                '  "question_analysis": {\n'
                '    "theme": "<core theme or topic in 3-8 words>",\n'
                '    "key_requirements": ["<requirement 1>", "<requirement 2>", "<requirement 3>"],\n'
                '    "relevance_verdict": "<1-2 sentences on how well the essay addressed the question>"\n'
                "  },\n"
            )
            question_rules = (
                '- "question_analysis.key_requirements" should list 2-4 specific things the question expects\n'
                '- "question_analysis.relevance_verdict" must be honest — if the essay missed the point, say so clearly\n'
                "- Use the question context to sharpen content issue explanations (e.g. flag when the essay ignores the stated theme)\n"
            )

        user_message = (
            "Mark the following PSLE English Continuous Writing composition.\n\n"
            + question_block
            + 'ESSAY:\n"""\n' + essay + '\n"""\n\n'
            "Respond with a JSON object in this EXACT schema - no other text:\n"
            "{\n"
            + question_schema
            + '  "scores": {\n'
            '    "content": <integer 0-18>,\n'
            '    "language": <integer 0-18>,\n'
            '    "total": <integer 0-36>\n'
            "  },\n"
            '  "issues": [\n'
            "    {\n"
            '      "snippet": "<exact short phrase from the essay, 3-15 words>",\n'
            '      "type": "<one of: logic | flow | language | content>",\n'
            '      "explanation": "<1-2 sentences explaining the problem>",\n'
            '      "suggestion": "<concrete corrected text or advice>",\n'
            '      "severity": "<one of: high | medium | low>"\n'
            "    }\n"
            "  ],\n"
            '  "strengths": ["<strength 1>", "<strength 2>"],\n'
            '  "improvements": ["<key improvement 1>", "<key improvement 2>"],\n'
            '  "overall_comment": "<2-3 sentence holistic comment a teacher would write>"\n'
            "}\n\n"
            "Rules:\n"
            + question_rules
            + '- "snippet" must be a verbatim excerpt from the essay (used for text highlighting)\n'
            '- "total" must equal "content" + "language"\n'
            "- Include 3-8 issues; do not flag trivial issues with severity low unless the essay is otherwise strong\n"
            '- "strengths" should have 2-4 items; "improvements" should have 2-4 items\n'
            "- If the essay is very short (under 100 words), set content score <= 8 and note underdevelopment"
        )

        raw = gemini_request(
            system=MARKING_SYSTEM_PROMPT,
            parts=[{"text": user_message}],
        )

        result = extract_json(raw)
        return jsonify({"success": True, "result": result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
