import numpy as np

# List of questions
questions_dict = {
    "Academic Pressure & Performance": [
        "Do you often feel overwhelmed by your academic workload?",
        "Do you struggle to manage your time effectively for studying?",
        "Do you feel anxious before exams or major assignments?",
        "Do you believe academic stress negatively affects your mental health?",
        "Do you find it difficult to stay motivated in your studies?",
        "Do you feel pressure from family or peers to excel academically?",
        "Have you ever considered dropping a course due to stress or workload?",
        "Do you feel confident in your ability to meet academic expectations?"
    ],
    "Social Life & Belonging": [
        "Do you feel comfortable making new friends at college?",
        "Do you often feel lonely or isolated in your college environment?",
        "Do you struggle with social anxiety in group settings?",
        "Do you feel supported by your friends when facing challenges?",
        "Do you feel like you truly belong in your college community?",
        "Have you avoided social events due to stress or anxiety?",
        "Do you feel comfortable expressing your thoughts and opinions among peers?",
        "Do you feel pressure to fit into certain social groups?"
    ],
    "Financial Stress": [
        "Does financial stress impact your ability to focus on academics?",
        "Do you worry about paying for tuition, rent, or other expenses?",
        "Do you feel you have enough financial support to complete your degree?",
        "Have you ever skipped meals or essential purchases due to money concerns?",
        "Do you feel pressure to work more hours than you can handle while studying?",
        "Are you concerned about student loan debt affecting your future?",
        "Do you believe financial stress is negatively affecting your mental health?"
    ],
    "Sleep & Physical Health": [
        "Do you often feel fatigued during the day due to lack of sleep?",
        "Do you have trouble falling or staying asleep due to stress?",
        "Do you experience frequent headaches, stomach issues, or muscle tension?",
        "Do you feel physically exhausted even after getting enough sleep?",
        "Do you engage in regular physical activity to maintain your health?",
        "Do you feel like your diet is negatively affecting your energy levels?",
        "Have you experienced unexplained changes in weight due to stress?",
        "Do you believe poor sleep is affecting your academic performance?"
    ],
    "Mental Health & Well-being": [
        "Do you frequently feel anxious or overwhelmed?",
        "Do you have difficulty managing your emotions during stressful situations?",
        "Do you feel like you have someone to talk to when feeling down?",
        "Have you ever avoided responsibilities due to mental health struggles?",
        "Do you feel a lack of motivation in areas of life that once excited you?",
        "Do you find yourself withdrawing from social interactions more often?",
        "Do you feel that your mental health is preventing you from achieving your goals?",
        "Have you ever felt like your stress levels were unmanageable?"
    ],
    "Academics & Future Preparation": [
        "Do you feel your degree is preparing you well for your career?",
        "Do you feel uncertain about your future career path?",
        "Do you feel stressed about life after graduation?",
        "Are you confident in your ability to find a job after graduation?",
        "Do you believe your college provides enough career support services?",
        "Do you feel you are developing the necessary skills for your desired career?",
        "Do you feel pressure to have your future fully planned out?",
        "Do you feel your current education aligns with your long-term goals?"
    ]
}


def get_random_close_questions():
    selected_questions = set()

    # Step 1: Pick one random question from each category
    for questions in questions_dict.values():
        selected_questions.add(np.random.choice(questions))

    # Step 2: If we have fewer than 10, randomly pick additional unique questions
    all_questions_pool = [q for questions in questions_dict.values() for q in questions]

    remaining_questions = list(set(all_questions_pool) - selected_questions)  # Exclude already chosen ones

    while len(selected_questions) < 6 and remaining_questions:
        chosen_question = np.random.choice(remaining_questions)
        selected_questions.add(chosen_question)
        remaining_questions.remove(chosen_question)  # Remove chosen question to avoid repetition

    return list(selected_questions)

# Example usage
random_questions = get_random_close_questions()
for question in random_questions:
    print(question)
