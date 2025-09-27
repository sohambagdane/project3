import numpy as np  

# Categorized questions
questions_dict = {
    "Academic Pressure & Performance": [
        "What aspects of your academic workload cause you the most stress?",
        "How do you handle pressure when facing deadlines or exams?",
        "What study habits have worked best for you, and what challenges do you face in maintaining them?",
        "How has your academic performance affected your self-esteem or mental health?",
        "How do you feel when comparing your academic progress to your peers?",
        "What support systems do you rely on when struggling with academics?"
    ],
    "Social Life & Belonging": [
        "What factors make you feel like you belong (or don’t belong) in your college community?",
        "How do social interactions impact your mental well-being?",
        "How do you cope with feelings of loneliness or social isolation?",
        "What role do friendships and relationships play in your overall happiness?",
        "How comfortable are you expressing your thoughts and emotions with others?"
    ],
    "Financial Stress": [
        "What financial concerns cause you the most stress, and how do you manage them?",
        "How does financial stress affect your daily life and mental health?",
        "What strategies have you tried to balance work, studies, and personal expenses?",
        "Have financial concerns ever affected your academic performance or future aspirations?"
    ],
    "Sleep & Physical Health": [
        "What factors influence your sleep quality, and how do they impact your mood?",
        "How do you feel when your sleep schedule is disrupted?",
        "How does your physical health impact your mental well-being?",
        "What lifestyle habits do you find helpful in maintaining both physical and mental health?"
    ],
    "Motivation & Goals": [
        "What drives you to achieve your academic and personal goals?",
        "How do you stay motivated when facing challenges in your studies or life?",
        "What setbacks have impacted your motivation, and how did you handle them?",
        "How do you define success, and how does this definition impact your mental well-being?"
    ],
    "General Well-being & Coping Mechanisms": [
        "What are the biggest stressors in your life right now, and how are you dealing with them?",
        "How do you typically manage stress when it becomes overwhelming?",
        "What activities or habits help you feel calm and centered?",
        "How do you recognize when your mental health is declining, and what steps do you take?"
    ],
    "Resources & Support": [
        "What kind of support do you wish was more available to students for mental health?",
        "How do you feel about seeking help from counselors, therapists, or mental health professionals?",
        "What barriers (if any) prevent you from seeking help when you need it?",
        "How well do you think your college supports students struggling with mental health challenges?"
    ]
}


def get_random_open_questions():
    selected_questions = set()

    # Step 1: Pick one random question from each category
    for questions in questions_dict.values():
        selected_questions.add(np.random.choice(questions))

    # Step 2: If we have fewer than 10, randomly pick additional unique questions
    all_questions_pool = [q for questions in questions_dict.values() for q in questions]
    
    remaining_questions = list(set(all_questions_pool) - selected_questions)  # Exclude already chosen ones

    while len(selected_questions) < 10 and remaining_questions:
        chosen_question = np.random.choice(remaining_questions)
        selected_questions.add(chosen_question)
        remaining_questions.remove(chosen_question)  # Remove chosen question to avoid repetition

    return list(selected_questions)

# Example usage
random_questions = get_random_open_questions()
for question in random_questions:
    print(question)
