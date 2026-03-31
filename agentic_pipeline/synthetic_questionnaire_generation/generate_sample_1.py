import csv
import json

personas = [
    # ── RELAXED (4) ──────────────────────────────────────────────────────────
    {
        "persona_id": "test_001",
        "persona_description": "22-year-old female student, mentally healthy, opened the app out of curiosity after a lecture on mental health.",
        "expected_flag": "relaxed",
        "expected_total_score": 0,
        "answers": {
            "q_gender": "female",
            "q_age": 22,
            "q_occupation": "student",
            "q_has_mental_concern": "fine",
            "q_phq2_1": 0,
            "q_phq2_2": 0,
            "q_gad2_1": 0,
            "q_gad2_2": 0,
            "q_daily_impact": "not_at_all",
            "q_prior_help": "never",
            "q_app_goal": ["just_curious", "info"]
        }
        # score: fine(0*3=0) + PHQ2(0+0=0<3→0) + GAD2(0+0=0<3→0) + not_at_all(0) + never(0) = 0
    },
    {
        "persona_id": "test_002",
        "persona_description": "46-year-old employed male who follows wellness trends and wants to learn more about mental health topics for self-development.",
        "expected_flag": "relaxed",
        "expected_total_score": 1,
        "answers": {
            "q_gender": "male",
            "q_age": 46,
            "q_occupation": "employed",
            "q_has_mental_concern": "fine",
            "q_phq2_1": 1,
            "q_phq2_2": 0,
            "q_gad2_1": 1,
            "q_gad2_2": 0,
            "q_daily_impact": "slightly",
            "q_prior_help": "never",
            "q_app_goal": ["info", "just_curious"]
        }
        # score: fine(0) + PHQ2(1+0=1<3→0) + GAD2(1+0=1<3→0) + slightly(1) + never(0) = 1
    },
    {
        "persona_id": "test_003",
        "persona_description": "58-year-old retired woman, generally content, previously did short-term therapy years ago, now exploring the app to stay informed.",
        "expected_flag": "relaxed",
        "expected_total_score": 1,
        "answers": {
            "q_gender": "female",
            "q_age": 58,
            "q_occupation": "retired",
            "q_has_mental_concern": "fine",
            "q_phq2_1": 0,
            "q_phq2_2": 1,
            "q_gad2_1": 0,
            "q_gad2_2": 0,
            "q_daily_impact": "not_at_all",
            "q_prior_help": "in_past",
            "q_diagnosis_known": "no",
            "q_app_goal": ["info"]
        }
        # score: fine(0) + PHQ2(0+1=1<3→0) + GAD2(0+0=0<3→0) + not_at_all(0) + in_past(1) = 1
    },
    {
        "persona_id": "test_004",
        "persona_description": "15-year-old student who prefers not to disclose gender; mildly stressed about school exams but considers it normal, had brief counseling at school before.",
        "expected_flag": "relaxed",
        "expected_total_score": 2,
        "answers": {
            "q_gender": "prefer_not",
            "q_age": 15,
            "q_occupation": "student",
            "q_has_mental_concern": "fine",
            "q_phq2_1": 1,
            "q_phq2_2": 1,
            "q_gad2_1": 0,
            "q_gad2_2": 1,
            "q_daily_impact": "slightly",
            "q_prior_help": "in_past",
            "q_diagnosis_known": "no",
            "q_app_goal": ["understand_myself", "just_curious"]
        }
        # score: fine(0) + PHQ2(1+1=2<3→0) + GAD2(0+1=1<3→0) + slightly(1) + in_past(1) = 2
    },

    # ── CONCERNED (7) ────────────────────────────────────────────────────────
    {
        "persona_id": "test_005",
        "persona_description": "29-year-old unemployed woman with persistent low mood and hopelessness for over 6 months; previously diagnosed with depression.",
        "expected_flag": "concerned",
        "expected_total_score": 16,
        "answers": {
            "q_gender": "female",
            "q_age": 29,
            "q_occupation": "unemployed",
            "q_has_mental_concern": "strong_concern",
            "q_concern_areas": ["depression", "sleep"],
            "q_phq2_1": 3,
            "q_phq2_2": 3,
            "q_gad2_1": 1,
            "q_gad2_2": 0,
            "q_daily_impact": "significantly",
            "q_duration": "more_6m",
            "q_prior_help": "in_past",
            "q_diagnosis_known": "yes",
            "q_app_goal": ["vent", "coping_tips", "find_specialist"]
        }
        # score: strong_concern(3*3=9) + PHQ2(3+3=6≥3→3) + GAD2(1+0=1<3→0) + significantly(3) + in_past(1) = 16
    },
    {
        "persona_id": "test_006",
        "persona_description": "42-year-old employed man experiencing gradual loss of enjoyment in work and hobbies, trouble concentrating, and disrupted sleep for the past 2 months.",
        "expected_flag": "concerned",
        "expected_total_score": 11,
        "answers": {
            "q_gender": "male",
            "q_age": 42,
            "q_occupation": "employed",
            "q_has_mental_concern": "moderate_concern",
            "q_concern_areas": ["depression", "concentration", "sleep"],
            "q_phq2_1": 2,
            "q_phq2_2": 3,
            "q_gad2_1": 1,
            "q_gad2_2": 1,
            "q_daily_impact": "moderately",
            "q_duration": "1m_6m",
            "q_prior_help": "never",
            "q_app_goal": ["understand_myself", "coping_tips"]
        }
        # score: moderate_concern(2*3=6) + PHQ2(2+3=5≥3→3) + GAD2(1+1=2<3→0) + moderately(2) + never(0) = 11
    },
    {
        "persona_id": "test_007",
        "persona_description": "24-year-old female student with severe generalized anxiety and OCD rituals that significantly interfere with academic performance; currently in therapy.",
        "expected_flag": "concerned",
        "expected_total_score": 16,
        "answers": {
            "q_gender": "female",
            "q_age": 24,
            "q_occupation": "student",
            "q_has_mental_concern": "strong_concern",
            "q_concern_areas": ["anxiety", "ocd", "panic"],
            "q_phq2_1": 1,
            "q_phq2_2": 1,
            "q_gad2_1": 3,
            "q_gad2_2": 3,
            "q_daily_impact": "significantly",
            "q_duration": "1m_6m",
            "q_prior_help": "currently",
            "q_diagnosis_known": "yes",
            "q_app_goal": ["coping_tips", "understand_myself"]
        }
        # score: strong_concern(3*3=9) + PHQ2(1+1=2<3→0) + GAD2(3+3=6≥3→3) + significantly(3) + currently(1) = 16
    },
    {
        "persona_id": "test_008",
        "persona_description": "36-year-old employed man dealing with persistent intrusive thoughts and checking compulsions that take up several hours per day.",
        "expected_flag": "concerned",
        "expected_total_score": 11,
        "answers": {
            "q_gender": "male",
            "q_age": 36,
            "q_occupation": "employed",
            "q_has_mental_concern": "moderate_concern",
            "q_concern_areas": ["anxiety", "ocd"],
            "q_phq2_1": 1,
            "q_phq2_2": 0,
            "q_gad2_1": 2,
            "q_gad2_2": 2,
            "q_daily_impact": "moderately",
            "q_duration": "more_6m",
            "q_prior_help": "never",
            "q_app_goal": ["understand_myself", "coping_tips"]
        }
        # score: moderate_concern(2*3=6) + PHQ2(1+0=1<3→0) + GAD2(2+2=4≥3→3) + moderately(2) + never(0) = 11
    },
    {
        "persona_id": "test_009",
        "persona_description": "17-year-old male student struggling with sustained attention, frequently losing track in class and forgetting tasks; teachers have flagged academic decline.",
        "expected_flag": "concerned",
        "expected_total_score": 12,
        "answers": {
            "q_gender": "male",
            "q_age": 17,
            "q_occupation": "student",
            "q_has_mental_concern": "moderate_concern",
            "q_concern_areas": ["concentration", "relationships"],
            "q_phq2_1": 1,
            "q_phq2_2": 2,
            "q_gad2_1": 1,
            "q_gad2_2": 1,
            "q_daily_impact": "significantly",
            "q_duration": "more_6m",
            "q_prior_help": "never",
            "q_app_goal": ["understand_myself", "coping_tips"]
        }
        # score: moderate_concern(2*3=6) + PHQ2(1+2=3≥3→3) + GAD2(1+1=2<3→0) + significantly(3) + never(0) = 12
    },
    {
        "persona_id": "test_010",
        "persona_description": "31-year-old employed woman who has struggled with focus and task initiation throughout her career; previously evaluated but not formally diagnosed.",
        "expected_flag": "concerned",
        "expected_total_score": 9,
        "answers": {
            "q_gender": "female",
            "q_age": 31,
            "q_occupation": "employed",
            "q_has_mental_concern": "moderate_concern",
            "q_concern_areas": ["concentration", "sleep", "anxiety"],
            "q_phq2_1": 1,
            "q_phq2_2": 1,
            "q_gad2_1": 2,
            "q_gad2_2": 0,
            "q_daily_impact": "moderately",
            "q_duration": "more_6m",
            "q_prior_help": "in_past",
            "q_diagnosis_known": "no",
            "q_app_goal": ["understand_myself", "coping_tips", "find_specialist"]
        }
        # score: moderate_concern(2*3=6) + PHQ2(1+1=2<3→0) + GAD2(2+0=2<3→0) + moderately(2) + in_past(1) = 9
    },
    {
        "persona_id": "test_011",
        "persona_description": "26-year-old employed male who has always found social interactions exhausting and confusing; reports difficulty forming relationships and recent low mood.",
        "expected_flag": "concerned",
        "expected_total_score": 12,
        "answers": {
            "q_gender": "male",
            "q_age": 26,
            "q_occupation": "employed",
            "q_has_mental_concern": "moderate_concern",
            "q_concern_areas": ["relationships", "concentration", "mood_swings"],
            "q_phq2_1": 1,
            "q_phq2_2": 2,
            "q_gad2_1": 1,
            "q_gad2_2": 1,
            "q_daily_impact": "moderately",
            "q_duration": "more_6m",
            "q_prior_help": "in_past",
            "q_diagnosis_known": "no",
            "q_app_goal": ["understand_myself", "vent"]
        }
        # score: moderate_concern(2*3=6) + PHQ2(1+2=3≥3→3) + GAD2(1+1=2<3→0) + moderately(2) + in_past(1) = 12
    },

    # ── URGENT (2) ───────────────────────────────────────────────────────────
    {
        "persona_id": "test_012",
        "persona_description": "34-year-old employed woman with trauma history; reports severe mood swings, relationship breakdowns, persistent nightmares, and passive thoughts of self-harm.",
        "expected_flag": "urgent",
        "expected_total_score": 24,
        "answers": {
            "q_gender": "female",
            "q_age": 34,
            "q_occupation": "employed",
            "q_has_mental_concern": "strong_concern",
            "q_concern_areas": ["mood_swings", "relationships", "sleep", "self_harm"],
            "q_phq2_1": 2,
            "q_phq2_2": 2,
            "q_gad2_1": 2,
            "q_gad2_2": 2,
            "q_daily_impact": "severely",
            "q_duration": "more_6m",
            "q_prior_help": "in_past",
            "q_diagnosis_known": "yes",
            "q_app_goal": ["find_specialist", "coping_tips"]
        }
        # flag: urgent (self_harm selected — overrides score)
        # underlying score: strong_concern(9) + PHQ2(4≥3→3) + GAD2(4≥3→3) + severely(4) + in_past(1) = 20
    },
    {
        "persona_id": "test_013",
        "persona_description": "20-year-old female student experiencing severe depression with hopelessness and recent self-harm ideation; has not previously sought professional help.",
        "expected_flag": "urgent",
        "expected_total_score": 19,
        "answers": {
            "q_gender": "female",
            "q_age": 20,
            "q_occupation": "student",
            "q_has_mental_concern": "strong_concern",
            "q_concern_areas": ["depression", "sleep", "self_harm"],
            "q_phq2_1": 3,
            "q_phq2_2": 3,
            "q_gad2_1": 2,
            "q_gad2_2": 1,
            "q_daily_impact": "severely",
            "q_duration": "1m_6m",
            "q_prior_help": "never",
            "q_app_goal": ["find_specialist", "vent"]
        }
        # flag: urgent (self_harm selected — overrides score)
        # underlying score: strong_concern(9) + PHQ2(6≥3→3) + GAD2(3≥3→3) + severely(4) + never(0) = 19
    },

    # ── EDGE CASES (2) ───────────────────────────────────────────────────────
    {
        "persona_id": "test_014",
        "persona_description": "33-year-old employed man who reports being 'fine' but acknowledges moderate emotional interference with daily tasks in recent weeks; borderline relaxed.",
        "expected_flag": "relaxed",
        "expected_total_score": 2,
        "answers": {
            "q_gender": "male",
            "q_age": 33,
            "q_occupation": "employed",
            "q_has_mental_concern": "fine",
            "q_phq2_1": 1,
            "q_phq2_2": 1,
            "q_gad2_1": 0,
            "q_gad2_2": 0,
            "q_daily_impact": "moderately",
            "q_duration": "less_2w",
            "q_prior_help": "never",
            "q_app_goal": ["just_curious", "understand_myself"]
        }
        # score: fine(0) + PHQ2(1+1=2<3→0) + GAD2(0+0=0<3→0) + moderately(2) + never(0) = 2  → relaxed
    },
    {
        "persona_id": "test_015",
        "persona_description": "48-year-old unemployed person (prefers not to disclose gender) who insists they feel 'fine' but PHQ-2 scores indicate frequent depressive symptoms; borderline concerned.",
        "expected_flag": "concerned",
        "expected_total_score": 3,
        "answers": {
            "q_gender": "prefer_not",
            "q_age": 48,
            "q_occupation": "unemployed",
            "q_has_mental_concern": "fine",
            "q_phq2_1": 2,
            "q_phq2_2": 1,
            "q_gad2_1": 0,
            "q_gad2_2": 0,
            "q_daily_impact": "not_at_all",
            "q_prior_help": "never",
            "q_app_goal": ["just_curious"]
        }
        # score: fine(0) + PHQ2(2+1=3≥3→3) + GAD2(0+0=0<3→0) + not_at_all(0) + never(0) = 3  → concerned
    },
]

output_path = "/Users/elenaalgazinova/Desktop/ALL_COMMITS/NLP4MentalHealth/agentic_pipeline/synthetic_questionnaire_generation/questionnaire_sample_1.csv"

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(["persona_id", "persona_description", "expected_flag", "expected_total_score", "answers_json"])
    for p in personas:
        writer.writerow([
            p["persona_id"],
            p["persona_description"],
            p["expected_flag"],
            p["expected_total_score"],
            json.dumps(p["answers"], ensure_ascii=False)
        ])

print(f"Written {len(personas)} rows to {output_path}")

# Validation table
print()
print("| persona_id | expected_flag | expected_total_score | calculation_trace |")
print("|------------|--------------|----------------------|-------------------|")
traces = {
    "test_001": "fine(0×3=0) + PHQ2(0+0<3→0) + GAD2(0+0<3→0) + not_at_all(0) + never(0) = 0",
    "test_002": "fine(0×3=0) + PHQ2(1+0=1<3→0) + GAD2(1+0=1<3→0) + slightly(1) + never(0) = 1",
    "test_003": "fine(0×3=0) + PHQ2(0+1=1<3→0) + GAD2(0+0=0<3→0) + not_at_all(0) + in_past(1) = 1",
    "test_004": "fine(0×3=0) + PHQ2(1+1=2<3→0) + GAD2(0+1=1<3→0) + slightly(1) + in_past(1) = 2",
    "test_005": "strong_concern(3×3=9) + PHQ2(3+3=6≥3→3) + GAD2(1+0=1<3→0) + significantly(3) + in_past(1) = 16",
    "test_006": "moderate_concern(2×3=6) + PHQ2(2+3=5≥3→3) + GAD2(1+1=2<3→0) + moderately(2) + never(0) = 11",
    "test_007": "strong_concern(3×3=9) + PHQ2(1+1=2<3→0) + GAD2(3+3=6≥3→3) + significantly(3) + currently(1) = 16",
    "test_008": "moderate_concern(2×3=6) + PHQ2(1+0=1<3→0) + GAD2(2+2=4≥3→3) + moderately(2) + never(0) = 11",
    "test_009": "moderate_concern(2×3=6) + PHQ2(1+2=3≥3→3) + GAD2(1+1=2<3→0) + significantly(3) + never(0) = 12",
    "test_010": "moderate_concern(2×3=6) + PHQ2(1+1=2<3→0) + GAD2(2+0=2<3→0) + moderately(2) + in_past(1) = 9",
    "test_011": "moderate_concern(2×3=6) + PHQ2(1+2=3≥3→3) + GAD2(1+1=2<3→0) + moderately(2) + in_past(1) = 12",
    "test_012": "URGENT(self_harm override); underlying: strong(9)+PHQ2(4≥3→3)+GAD2(4≥3→3)+severely(4)+in_past(1)=20",
    "test_013": "URGENT(self_harm override); underlying: strong(9)+PHQ2(6≥3→3)+GAD2(3≥3→3)+severely(4)+never(0)=19",
    "test_014": "fine(0×3=0) + PHQ2(1+1=2<3→0) + GAD2(0+0=0<3→0) + moderately(2) + never(0) = 2 [EDGE relaxed]",
    "test_015": "fine(0×3=0) + PHQ2(2+1=3≥3→3) + GAD2(0+0=0<3→0) + not_at_all(0) + never(0) = 3 [EDGE concerned]",
}
for p in personas:
    print(f"| {p['persona_id']} | {p['expected_flag']} | {p['expected_total_score']} | {traces[p['persona_id']]} |")
