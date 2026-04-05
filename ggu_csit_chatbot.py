# ============================================================
#  GGU CSIT Chatbot — Guru Ghasidas Vishwavidyalaya, Bilaspur
#  Department of Computer Science & Information Technology
#
#  Features:
#    - Loads data from ggu_data.json (keep both files together)
#    - NLTK-based NLP (tokenization + lemmatization)
#    - Fuzzy keyword matching (handles typos)
#    - Context memory (remembers last topic)
#    - Detailed course, faculty, and fee responses
#
#  Install:  pip install nltk colorama
#  Run:      python ggu_csit_chatbot.py
# ============================================================

import json
import os
import sys
import string
import random
import difflib
import nltk
import datetime
import textwrap

__version__ = "2.0.0"
__author__  = "GGU CSIT Dept."


from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from colorama import Fore, Style, init

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)

init(autoreset=True)
lemmatizer = WordNetLemmatizer()


# ─── Load Data ────────────────────────────────────────────────────────────────

def load_data() -> dict:
    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggu_data.json")
    if not os.path.exists(data_file):
        print(Fore.RED + "ERROR: ggu_data.json not found. "
              "Please keep it in the same folder as this script.")
        exit(1)
    with open(data_file, "r", encoding="utf-8") as f:
        return json.load(f)

DATA = load_data()


# ─── Intent Keyword Map ───────────────────────────────────────────────────────

INTENTS = {
    "greeting":    ["hi", "hello", "hey", "howdy", "greet", "morning", "afternoon", "evening", "namaste", "start"],
    "farewell":    ["bye", "goodbye", "exit", "quit", "later", "tata", "cya", "close", "end"],
    "thanks":      ["thanks", "thank", "appreciate", "grateful", "cheers", "helpful"],
    "about":       ["about", "overview", "history", "established", "founded", "what", "information",
                    "info", "tell", "ggu", "ggv", "university", "vishwavidyalaya", "department", "csit"],
    "courses":     ["course", "program", "programme", "degree", "branch", "stream", "study",
                    "offer", "available", "subject", "list", "all"],
    "mca":         ["mca", "master", "application"],
    "msc_cs":      ["msc", "m.sc"],
    "msc_it":      ["mscit", "m.sc.it", "information technology", "msc information"],
    "integrated":  ["integrated", "bsc", "b.sc", "undergraduate", "five"],
    "phd":         ["phd", "ph.d", "doctoral", "doctorate", "vret"],
    "help":        ["help", "menu", "option", "command", "what can", "guide", "assist", "support"],
    "compare":     ["compare", "difference", "vs", "versus", "better", "which", "choose",
                    "prefer", "between", "best", "suit"],
    "fees":        ["fee", "fees", "cost", "price", "tuition", "charge", "payment", "money",
                    "rupee", "hostel", "expense", "afford"],
    "admission":   ["admission", "admit", "apply", "application", "enroll", "join", "entrance",
                    "eligibility", "criteria", "requirement", "cuet", "gate", "process"],
    "faculty":     ["faculty", "professor", "teacher", "staff", "lecturer", "doctor", "instructor",
                    "who", "teach", "sir", "madam"],
    "professor":   ["babita", "santosh", "pushplata", "rajwant", "sushma", "akhilesh", "vikas",
                    "vineet", "prashant", "vivek", "abhishek", "amitesh", "majhi", "pujari",
                    "jaiswal", "shrivas", "pandey", "awasthi", "vaishnav", "sarthe", "patel", "jha", "rao"],
    "facilities":  ["facility", "facilities", "hostel", "library", "lab", "wifi", "internet",
                    "sport", "cafeteria", "infrastructure", "amenity", "medical", "bus", "transport", "campus"],
    "placement":   ["placement", "job", "package", "salary", "recruit", "recruiter", "hire",
                    "lpa", "company", "career", "placed", "opportunity"],
    "scholarship": ["scholarship", "financial", "aid", "waiver", "merit", "stipend",
                    "concession", "fund", "fellowship", "nsp", "free"],
    "research":    ["research", "publication", "conference", "paper", "project", "journal",
                    "innovation", "patent", "highlight"],
    "exam":        ["exam", "examination", "test", "internal", "assessment",
                    "marks", "pattern", "grade", "cgpa", "evaluation"],
    "location":    ["location", "address", "where", "place", "city", "situated", "koni",
                    "bilaspur", "chhattisgarh", "distance", "map"],
    "contact":     ["contact", "phone", "email", "number", "reach", "call", "mail", "website", "helpline"],
    "semester":    ["semester", "syllabus", "curriculum", "sem"],
}

SMALL_TALK = {
    "greeting": [
        "Namaste! 🙏  Welcome to the GGU CSIT Dept. chatbot.\n"
        "  Ask me about: courses · faculty · fees · admission · placement · research",
        "Hello! I'm the CSIT Dept. assistant at Guru Ghasidas Vishwavidyalaya.\n"
        "  What would you like to know?",
    ],
    "farewell": [
        "Goodbye! Best of luck with your studies! 🎓  — GGU CSIT Dept.",
        "See you! Feel free to return anytime. 😊",
    ],
    "thanks": [
        "You're welcome! Any other questions about CSIT at GGU?",
        "Happy to help! Anything else you'd like to know?",
    ],
    "unknown": [
        "I'm not sure about that. Try asking about:\n"
        "  courses · fees · faculty · admission · placement · scholarship · research",
        "Could you rephrase? Example:\n"
        "  'Tell me about MCA' · 'Who is Dr. Shrivas?' · 'What are the fees?'",
    ],
}


# ─── NLP & Fuzzy Matching ────────────────────────────────────────────────────

def preprocess(text: str) -> list:
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation]
    return [lemmatizer.lemmatize(t) for t in tokens]


ALL_KEYWORDS = {kw: intent for intent, kws in INTENTS.items() for kw in kws}


def fuzzy_match(token: str, cutoff: float = 0.82):
    matches = difflib.get_close_matches(token, ALL_KEYWORDS.keys(), n=1, cutoff=cutoff)
    return ALL_KEYWORDS[matches[0]] if matches else None


def detect_intent(tokens: list) -> tuple:
    scores = {intent: 0 for intent in INTENTS}
    matched_professor = None

    for token in tokens:
        for intent, keywords in INTENTS.items():
            if token in keywords:
                scores[intent] += 2
                if intent == "professor":
                    matched_professor = token
        fuzzy = fuzzy_match(token)
        if fuzzy:
            scores[fuzzy] += 1

    # Detect specific course from text
    lower = " ".join(tokens)
    matched_course = None
    if "mca" in lower or "master of computer application" in lower:
        matched_course = "MCA (Master of Computer Applications)"
    elif "integrated" in lower or "bsc" in lower or "b.sc" in lower:
        matched_course = "Integrated UG/PG (B.Sc. + M.Sc.)"
    elif "msc it" in lower or "m.sc it" in lower or ("information" in lower and "technology" in lower):
        matched_course = "M.Sc. (Information Technology)"
    elif "msc cs" in lower or (("msc" in lower or "m.sc" in lower) and "computer" in lower):
        matched_course = "M.Sc. (Computer Science)"
    elif "phd" in lower or "ph.d" in lower or "doctoral" in lower:
        matched_course = "Ph.D. (Computer Science / IT)"

    best = max(scores, key=scores.get)
    sub  = matched_professor or matched_course
    return (best if scores[best] > 0 else "unknown", sub)


# ─── Context Memory ───────────────────────────────────────────────────────────

context = {
    "last_intent":  None,
    "last_course":  None,
    "last_faculty": None,
}

# ─── Session History & Feedback ──────────────────────────────────────────────

session_history: list[dict] = []   # [{turn, user, intent, bot}, …]
_turn_counter = 0


def _log(user_msg: str, intent: str, bot_reply: str) -> None:
    global _turn_counter
    _turn_counter += 1
    session_history.append({
        "turn":   _turn_counter,
        "time":   datetime.datetime.now().strftime("%H:%M:%S"),
        "user":   user_msg,
        "intent": intent,
        "bot":    bot_reply,
    })


def show_history() -> str:
    if not session_history:
        return "📜  No conversation history yet."
    lines = ["📜  Conversation History (this session):\n"]
    for entry in session_history:
        lines.append(f"  [{entry['turn']:02d}] {entry['time']}  You: {entry['user']}")
        # Show only first line of bot reply to keep it scannable
        first_line = entry["bot"].splitlines()[0]
        lines.append(f"       Bot: {first_line}")
    return "\n".join(lines)


def session_summary() -> str:
    if not session_history:
        return "Nothing to summarise yet."
    intents_used = [e["intent"] for e in session_history if e["intent"] != "unknown"]
    unique = list(dict.fromkeys(intents_used))   # preserve order, deduplicate
    lines = [
        "📊  Session Summary",
        f"  Turns        : {_turn_counter}",
        f"  Topics asked : {', '.join(unique) if unique else 'none'}",
        f"  Started at   : {session_history[0]['time']}",
        f"  Now          : {datetime.datetime.now().strftime('%H:%M:%S')}",
    ]
    return "\n".join(lines)


# ─── Course Comparison ───────────────────────────────────────────────────────

_COMPARE_KEYS = ("duration", "intake", "eligibility", "fee_per_semester",
                 "total_approx_fee", "fellowship")
_COMPARE_LABELS = {
    "duration":         "Duration",
    "intake":           "Intake",
    "eligibility":      "Eligibility",
    "fee_per_semester": "Fee / Semester",
    "total_approx_fee": "Total Fee",
    "fellowship":       "Fellowship",
}


def compare_courses(tokens: list) -> str | None:
    """Return a side-by-side comparison if ≥2 course names are found in tokens."""
    found = []
    lower = " ".join(tokens)
    mapping = {
        "mca":        "MCA (Master of Computer Applications)",
        "msc cs":     "M.Sc. (Computer Science)",
        "msc it":     "M.Sc. (Information Technology)",
        "integrated": "Integrated UG/PG (B.Sc. + M.Sc.)",
        "phd":        "Ph.D. (Computer Science / IT)",
        "ph.d":       "Ph.D. (Computer Science / IT)",
    }
    for kw, full in mapping.items():
        if kw in lower and full not in found:
            found.append(full)
    if len(found) < 2:
        return None

    courses_data = DATA["courses"]
    col_w = 26
    header = f"  {'Attribute':<20}" + "".join(f"  {n[:col_w]:<{col_w}}" for n in found)
    sep    = "  " + "─" * (20 + (col_w + 2) * len(found))
    lines  = [f"⚖️   Course Comparison\n", header, sep]
    for key in _COMPARE_KEYS:
        label = _COMPARE_LABELS[key]
        row   = f"  {label:<20}"
        for name in found:
            val = courses_data.get(name, {}).get(key, "—")
            row += f"  {str(val)[:col_w]:<{col_w}}"
        lines.append(row)
    lines.append(sep)
    lines.append(f"\n  💡  Ask 'Tell me about {found[0]}' for full details.")
    return "\n".join(lines)


# ─── Quick-Command Shortcuts ─────────────────────────────────────────────────

QUICK_COMMANDS = {
    "/help":    "help",
    "/courses": "courses",
    "/fees":    "fees",
    "/faculty": "faculty",
    "/admit":   "admission",
    "/place":   "placement",
    "/research":"research",
    "/contact": "contact",
    "/history": "history",
    "/summary": "summary",
    "/about":   "about",
    "/exam":    "exam",
}

HELP_TEXT = textwrap.dedent("""\
    🤖  GGU CSIT Chatbot — Help

    NATURAL LANGUAGE  (just type, typos OK)
      'Tell me about MCA'
      'What are the fees?'
      'Who is Dr. Shrivas?'
      'MCA vs Integrated comparison'
      'Ph.D admission process'
      'What is the exam pattern?'
      'MCA syllabus'

    QUICK COMMANDS
      /courses   — list all programmes
      /fees      — fee structure
      /faculty   — faculty list
      /admit     — admission process
      /place     — placements
      /research  — research highlights
      /contact   — contact info
      /history   — this session's Q&A
      /summary   — session statistics
      /about     — about GGU CSIT
      /exam      — exam pattern
      /help      — show this help

    TIPS
      • Ask about a course, then just say 'fees?' or 'syllabus' — context is remembered.
      • Use a professor's last name: 'Tell me about Majhi'
      • Compare programmes: 'MCA vs Ph.D'
      • Type 'bye' or 'exit' to quit.
""")


def save_log_to_file() -> str:
    """Save session transcript to a dated .txt file next to the script."""
    if not session_history:
        return ""
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"ggu_chat_log_{ts}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"GGU CSIT Chatbot — Session Log  ({ts})\n")
        fh.write("=" * 60 + "\n\n")
        for entry in session_history:
            fh.write(f"[{entry['turn']:02d}] {entry['time']}\n")
            fh.write(f"You : {entry['user']}\n")
            fh.write(f"Bot : {entry['bot']}\n\n")
        fh.write(session_summary())
    return path


def _spell_hint(user_input: str) -> str:
    """Return a suggestion line if a close keyword match exists."""
    tokens = preprocess(user_input)
    suggestions = set()
    for tok in tokens:
        if len(tok) < 3:
            continue
        close = difflib.get_close_matches(tok, ALL_KEYWORDS.keys(), n=2, cutoff=0.75)
        for c in close:
            if c != tok:
                suggestions.add(c)
    if suggestions:
        return "  💡  Did you mean: " + " / ".join(sorted(suggestions)[:3]) + "?"
    return ""


# ─── Response Builder ─────────────────────────────────────────────────────────

def build_response(intent: str, sub) -> str:
    global context

    if intent in SMALL_TALK:
        return random.choice(SMALL_TALK[intent])

    d = DATA

    # ── About ───────────────────────────────────────────────────────────────
    if intent == "about":
        context["last_intent"] = "about"
        return (
            f"🏛️   {d['university']}\n"
            f"     {d['department']}\n\n"
            f"  Location  : {d['location']}\n"
            f"  Founded   : {d['established']}\n"
            f"  NAAC      : {d['naac']}\n"
            f"  Campus    : {d['campus_size']}\n"
            f"  Website   : {d['website']}\n\n"
            "  The CSIT dept. started with B.Sc. CS, added M.Sc. CS & IT (1996),\n"
            "  MCA (1998, AICTE approved), Integrated UG/PG, and Ph.D. programmes.\n"
            "  Faculty collaborate with universities across India and abroad."
        )

    # ── All courses ─────────────────────────────────────────────────────────
    if intent == "courses":
        context["last_intent"] = "courses"
        lines = ["🎓  Programmes offered by CSIT Dept., GGV:\n"]
        for prog, info in d["courses"].items():
            lines.append(
                f"  • {prog}\n"
                f"    Duration : {info['duration']} | Seats : {info.get('intake', 'N/A')}\n"
                f"    Fee/sem  : {info.get('fee_per_semester', 'N/A')}"
            )
        lines.append("\n  💡  Ask 'Tell me about MCA' or 'MCA syllabus' for full details.")
        return "\n".join(lines)

    # ── Specific course detail ──────────────────────────────────────────────
    course_map = {
        "mca":        "MCA (Master of Computer Applications)",
        "msc_cs":     "M.Sc. (Computer Science)",
        "msc_it":     "M.Sc. (Information Technology)",
        "integrated": "Integrated UG/PG (B.Sc. + M.Sc.)",
        "phd":        "Ph.D. (Computer Science / IT)",
    }
    course_key = course_map.get(intent) or (sub if sub and sub in d["courses"] else None)
    if not course_key and context["last_course"]:
        course_key = context["last_course"]

    if course_key and course_key in d["courses"]:
        context["last_course"]  = course_key
        context["last_intent"]  = "course_detail"
        c = d["courses"][course_key]
        lines = [f"📘  {course_key}\n"]
        lines.append(f"  Duration    : {c['duration']}")
        if "intake" in c:
            lines.append(f"  Intake      : {c['intake']}")
        lines.append(f"  Eligibility : {c['eligibility']}")
        if "exit_option" in c:
            lines.append(f"  Exit Option : {c['exit_option']}")
        lines.append(f"  Fee/Semester: {c.get('fee_per_semester', 'N/A')}")
        if "total_approx_fee" in c:
            lines.append(f"  Total Fee   : {c['total_approx_fee']}")
        if "fellowship" in c:
            lines.append(f"  Fellowship  : {c['fellowship']}")
        if "research_areas" in c:
            lines.append(f"  Research    : {', '.join(c['research_areas'])}")
        if "semesters" in c:
            lines.append("\n  📋  Semester-wise Subjects:")
            for sem, subjects in c["semesters"].items():
                lines.append(f"\n    {sem}:")
                for subj in subjects:
                    lines.append(f"      - {subj}")
        return "\n".join(lines)

    # ── Semester / syllabus ─────────────────────────────────────────────────
    if intent == "semester":
        if context["last_course"] and context["last_course"] in d["courses"]:
            c = d["courses"][context["last_course"]]
            if "semesters" in c:
                lines = [f"📋  Semester-wise Subjects — {context['last_course']}:\n"]
                for sem, subjects in c["semesters"].items():
                    lines.append(f"  {sem}:")
                    for subj in subjects:
                        lines.append(f"    - {subj}")
                return "\n".join(lines)
        return ("Please specify a course first. Example:\n"
                "  'MCA syllabus'  or  'Integrated programme semesters'")

    # ── Fees ────────────────────────────────────────────────────────────────
    if intent == "fees":
        context["last_intent"] = "fees"
        fee = d["fees"]
        lines = ["💰  Fee Structure — CSIT Dept., GGV:\n"]
        for prog, info in d["courses"].items():
            lines.append(
                f"  • {prog}\n"
                f"    Per Semester : {info.get('fee_per_semester', 'N/A')}\n"
                f"    Total        : {info.get('total_approx_fee', 'N/A')}"
            )
        lines.append(
            f"\n  🏠  Hostel:\n"
            f"    Boys  : {fee['hostel']['boys']}\n"
            f"    Girls : {fee['hostel']['girls']}\n"
            f"    Note  : {fee['hostel']['note']}"
        )
        lines.append(f"\n  ⚠️   {fee['note']}")
        return "\n".join(lines)

    # ── Admission ───────────────────────────────────────────────────────────
    if intent == "admission":
        context["last_intent"] = "admission"
        lines = ["📝  Admission Process — CSIT Dept., GGV:\n"]
        for prog, info in d["admission_process"].items():
            lines.append(f"  • {prog}\n    {info}")
        lines.append("\n  📌  Apply online at: www.ggu.ac.in (SAMARTH Portal)")
        return "\n".join(lines)

    # ── Faculty list ────────────────────────────────────────────────────────
    if intent == "faculty":
        context["last_intent"] = "faculty"
        lines = ["👨‍🏫  Teaching Faculty — CSIT Dept., GGV:\n"]
        for f in d["faculty"]:
            lines.append(f"  • {f['name']}  |  {f['designation']}")
            lines.append(f"    {f['specialization']}")
        lines.append("\n  💡  Ask 'Tell me about Dr. Babita Majhi' for full profile.")
        return "\n".join(lines)

    # ── Specific professor profile ──────────────────────────────────────────
    if intent == "professor" or (sub and isinstance(sub, str) and any(
        sub.lower() in f["name"].lower() for f in d["faculty"]
    )):
        query = sub or ""
        matched_f = None
        for f in d["faculty"]:
            if query.lower() in f["name"].lower():
                matched_f = f
                break
        if not matched_f and context["last_faculty"]:
            for f in d["faculty"]:
                if context["last_faculty"].lower() in f["name"].lower():
                    matched_f = f
                    break

        if matched_f:
            context["last_faculty"] = matched_f["name"]
            f = matched_f
            lines = [f"👤  {f['name']}\n"]
            lines.append(f"  Designation     : {f['designation']}")
            lines.append(f"  Qualification   : {f['qualification']}")
            lines.append(f"  Specialization  : {f['specialization']}")
            lines.append(f"  Subjects Taught : {', '.join(f['subjects_taught'])}")
            lines.append(f"  Email           : {f['email']}")
            if f.get("phone") and "N/A" not in f["phone"]:
                lines.append(f"  Phone           : {f['phone']}")
            if f.get("notable"):
                lines.append(f"  Notable         : {f['notable']}")
            if f.get("google_scholar"):
                lines.append(f"  Google Scholar  : {f['google_scholar']}")
            if f.get("orcid"):
                lines.append(f"  ORCID           : {f['orcid']}")
            if f.get("joined"):
                lines.append(f"  Joined GGV      : {f['joined']}")
            return "\n".join(lines)
        else:
            return ("I couldn't find that professor. Try using their last name.\n"
                    "  Example: 'Tell me about Majhi' or 'Who is Dr. Shrivas?'\n"
                    "  Type 'faculty' to see the full list.")

    # ── Facilities ──────────────────────────────────────────────────────────
    if intent == "facilities":
        lines = ["🏫  Campus & Departmental Facilities:\n"]
        for item in d["facilities"]:
            lines.append(f"  • {item}")
        return "\n".join(lines)

    # ── Placement ───────────────────────────────────────────────────────────
    if intent == "placement":
        p = d["placement"]
        return (
            "💼  Placement Highlights — GGV CSIT:\n\n"
            f"  UG 4-yr Median Package : {p['ug_4yr_median']}\n"
            f"  PG 2-yr Median Package : {p['pg_2yr_median']}\n"
            f"  Highest Package        : {p['highest']}\n"
            f"  UG Students Placed     : {p['students_placed_ug']}\n"
            f"  PG Students Placed     : {p['students_placed_pg']}\n"
            f"  Placement Rate         : {p['placement_rate']}\n"
            f"  Top Recruiters         : {', '.join(p['top_recruiters'])}\n\n"
            f"  📌  {p['placement_cell']}\n"
            f"  {p['note']}"
        )

    # ── Scholarship ─────────────────────────────────────────────────────────
    if intent == "scholarship":
        lines = ["🏅  Scholarships & Financial Aid:\n"]
        for item in d["scholarship"]:
            lines.append(f"  • {item}")
        lines.append("\n  📌  Apply via: scholarships.gov.in or through SAMARTH Portal.")
        return "\n".join(lines)

    # ── Research ────────────────────────────────────────────────────────────
    if intent == "research":
        r = d["research"]
        lines = [f"🔬  Research @ CSIT, GGV:\n\n  {r['summary']}\n"]
        lines.append(f"  Research Areas : {', '.join(r['areas'])}\n")
        lines.append("  Recent Highlights:")
        for h in r["recent_highlights"]:
            lines.append(f"    • {h}")
        return "\n".join(lines)

    # ── Exam pattern ────────────────────────────────────────────────────────
    if intent == "exam":
        return f"📋  Examination Pattern:\n\n  {d['exam_pattern']}"

    # ── Location ────────────────────────────────────────────────────────────
    if intent == "location":
        return (
            f"📍  {d['university']}\n"
            f"    {d['department']}\n"
            f"    {d['location']}\n\n"
            "    Located at Koni, approx. 5 km from Bilaspur city.\n"
            "    River Arpa runs parallel to the campus.\n"
            "    University buses connect Bilaspur city to Koni campus."
        )

    # ── Contact ─────────────────────────────────────────────────────────────
    if intent == "contact":
        return (
            "📞  Contact — GGV CSIT Dept.:\n\n"
            f"  Phone   : {d['phone']}\n"
            f"  Email   : {d['email']}\n"
            f"  Website : {d['website']}"
        )

    # ── Help ─────────────────────────────────────────────────────────────────
    if intent == "help":
        return HELP_TEXT

    # ── Compare (natural language: "MCA vs M.Sc CS", "which is better MCA or MCA IT") ──
    if intent == "compare":
        alias_map = {
            "mca":        "MCA (Master of Computer Applications)",
            "msc cs":     "M.Sc. (Computer Science)",
            "msc it":     "M.Sc. (Information Technology)",
            "integrated": "Integrated UG/PG (B.Sc. + M.Sc.)",
            "phd":        "Ph.D. (Computer Science / IT)",
            "ph.d":       "Ph.D. (Computer Science / IT)",
        }
        hint_lower = (sub or "").lower()
        candidates = []
        for alias, full in alias_map.items():
            if alias in hint_lower and full not in candidates:
                candidates.append(full)
        if len(candidates) < 2 and context.get("last_course"):
            if context["last_course"] not in candidates:
                candidates.insert(0, context["last_course"])
        if len(candidates) >= 2:
            result = compare_courses([a for a in alias_map if alias_map[a] in candidates])
            if result:
                return result
        return (
            "⚖️   To compare courses, mention two programme names.\n"
            "  Examples:\n"
            "    'MCA vs M.Sc CS'\n"
            "    'Compare Integrated and MCA'\n"
            "    'Ph.D vs MCA difference'"
        )

    return random.choice(SMALL_TALK["unknown"])


# ─── Main Loop ────────────────────────────────────────────────────────────────

def chat():
    # ── CLI flag: --help ────────────────────────────────────────────────────
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print(HELP_TEXT)
        return

    title  = "  GGU CSIT Department Chatbot  "
    sub    = "  Guru Ghasidas Vishwavidyalaya, Bilaspur  "
    border = "─" * max(len(title), len(sub))

    print(Fore.CYAN + f"\n┌{border}┐")
    print(Fore.CYAN + f"│{title.center(len(border))}│")
    print(Fore.CYAN + f"│{sub.center(len(border))}│")
    print(Fore.CYAN + f"└{border}┘")
    print(Fore.YELLOW + "\n  Topics : courses · faculty · fees · admission · placement")
    print(Fore.YELLOW + "           scholarship · research · exam pattern · contact")
    print(Fore.YELLOW + "  Cmds   : /help  /courses  /fees  /faculty  /history  /summary")
    print(Fore.YELLOW + "  Type 'bye' or 'exit' to quit.\n")

    while True:
        # ── Prompt ──────────────────────────────────────────────────────────
        try:
            user_input = input(Fore.GREEN + "You  : " + Style.RESET_ALL).strip()
        except (KeyboardInterrupt, EOFError):
            print(Fore.YELLOW + "\nGoodbye! 👋")
            break

        if not user_input:
            continue

        # ── Quick commands (/help, /fees, …) ────────────────────────────────
        cmd = user_input.lower().split()[0]
        if cmd in QUICK_COMMANDS:
            mapped = QUICK_COMMANDS[cmd]

            if mapped == "help":
                response = HELP_TEXT
                _log(user_input, "help", response)
                print(Fore.CYAN + f"\nBot  : {response}\n")
                continue

            if mapped == "history":
                response = show_history()
                _log(user_input, "history", response)
                print(Fore.CYAN + f"\nBot  : {response}\n")
                continue

            if mapped == "summary":
                response = session_summary()
                _log(user_input, "summary", response)
                print(Fore.CYAN + f"\nBot  : {response}\n")
                continue

            # Treat slash-command as a natural-language intent
            tokens          = preprocess(mapped)
            intent, sub_hit = detect_intent(tokens)
            response        = build_response(intent, sub_hit)
            _log(user_input, intent, response)
            print(Fore.CYAN + f"\nBot  : {response}\n" + Style.RESET_ALL)
            continue

        # ── Course comparison  (e.g. "MCA vs Ph.D") ─────────────────────────
        if any(kw in user_input.lower() for kw in ("vs", "versus", "compare", "difference between")):
            cmp_tokens = preprocess(user_input)
            cmp_result = compare_courses(cmp_tokens)
            if cmp_result:
                _log(user_input, "comparison", cmp_result)
                print(Fore.CYAN + f"\nBot  : {cmp_result}\n" + Style.RESET_ALL)
                continue

        # ── Inline feedback  ("good bot", "bad", "wrong", …) ────────────────
        low = user_input.lower()
        if any(kw in low for kw in ("good bot", "great", "awesome", "perfect", "nice", "well done")):
            response = "😊  Thank you for the kind words! Anything else I can help with?"
            _log(user_input, "feedback_positive", response)
            print(Fore.CYAN + f"\nBot  : {response}\n")
            continue
        if any(kw in low for kw in ("wrong", "incorrect", "bad bot", "not helpful", "useless")):
            response = ("😔  Sorry about that! Please rephrase your question or type /help\n"
                        "  to see example queries. Your feedback helps us improve.")
            _log(user_input, "feedback_negative", response)
            print(Fore.CYAN + f"\nBot  : {response}\n")
            continue

        # ── Normal NLP pipeline ─────────────────────────────────────────────
        tokens          = preprocess(user_input)
        intent, sub_hit = detect_intent(tokens)
        response        = build_response(intent, sub_hit)

        # ── Spell hint on unknown ────────────────────────────────────────────
        if intent == "unknown":
            hint = _spell_hint(user_input)
            if hint:
                response = response + "\n" + hint

        _log(user_input, intent, response)
        print(Fore.CYAN + f"\nBot  : {response}\n" + Style.RESET_ALL)

        # ── Exit ─────────────────────────────────────────────────────────────
        if intent == "farewell":
            # Offer to save log
            try:
                save_q = input(Fore.YELLOW +
                               "  Save session log to file? [y/N]: " +
                               Style.RESET_ALL).strip().lower()
            except (KeyboardInterrupt, EOFError):
                save_q = "n"
            if save_q == "y":
                path = save_log_to_file()
                if path:
                    print(Fore.GREEN + f"  ✅  Log saved → {path}")
                else:
                    print(Fore.YELLOW + "  Nothing to save.")
            print()
            break


if __name__ == "__main__":
    chat()
