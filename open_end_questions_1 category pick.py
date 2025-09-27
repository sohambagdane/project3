import numpy as np

# Categorized questions
questions_dict = {
    "Academic Pressure & Performance": [
        "How often do you feel stressed about your academic performance?",
        "Do you find it difficult to concentrate on your studies?",
        "Have you noticed a change in your study habits or motivation?",
        "Do you feel pressure to achieve high grades from your family or yourself?",
        "Do you compare yourself unfavorably to other students?",
        "Are you worried about failing exams or assignments?",
        "Do you have difficulty managing your time between studies and other activities?",
        "Do you ever feel like you're not smart enough or don't belong in college/university?",
        "How often do you procrastinate on academic tasks?",
        "Do you experience physical symptoms of stress related to academics (e.g., headaches, stomach aches)?",
        "Are you able to ask for help when you're struggling with coursework?",
        "Do you feel supported by your professors or academic advisors?",
        "Do you have access to the academic resources you need (e.g., tutoring, library)?",
        "How often do you feel overwhelmed by the amount of work you have to do?",
        "Do you feel prepared for exams and assignments?"
    ],
    "Social Life & Belonging": [
        "Do you feel like you belong in your school community?",
        "Do you have close friends at school?",
        "Do you feel comfortable socializing with your classmates?",
        "Have you experienced any bullying or harassment at school?",
        "Do you feel isolated or lonely at school?",
        "Are you involved in any extracurricular activities?",
        "Do you feel pressure to conform to certain social norms at school?",
        "Do you find it difficult to balance your social life with your studies?",
        "Do you feel accepted by your peers?",
        "Are you comfortable expressing your opinions and beliefs in class?"
    ],
    "Financial Stress": [
        "Are you worried about your finances as a student?",
        "Do you have enough money to cover your basic needs (e.g., food, housing, books)?",
        "Do you have to work long hours to support yourself financially?",
        "Does financial stress affect your academic performance or mental health?",
        "Are you aware of any financial aid or scholarship opportunities?"
    ],
    "Sleep & Physical Health": [
        "How many hours of sleep do you typically get each night?",
        "Do you feel rested when you wake up in the morning?",
        "Have you noticed any changes in your appetite or eating habits?",
        "Do you exercise regularly?",
        "Do you feel physically healthy?"
    ],
    "Motivation & Goals": [
        "Do you feel motivated to pursue your academic goals?",
        "Do you have a clear sense of what you want to achieve in your studies?",
        "Do you feel like your education is meaningful to you?",
        "Do you feel optimistic about your future?",
        "Are you able to set realistic goals for yourself?"
    ],
    "General Well-being": [
        "How would you rate your overall well-being?",
        "Do you feel hopeful about the future?",
        "Do you enjoy your time as a student?",
        "Do you feel like you have a good work-life balance?",
        "Do you take time for yourself to relax and recharge?"
    ],
    "Resources & Support": [
        "Are you aware of the mental health resources available to students at your school?",
        "Would you feel comfortable seeking help from a counselor or therapist?",
        "Do you know who to talk to if you're struggling with your mental health?",
        "Do you feel supported by the staff at your school?",
        "If you were struggling, would you know where to go for help on campus?"
    ]
}

def get_random_open_questions():
    all_questions = []
    for questions in questions_dict.values():
        all_questions.extend(np.random.choice(questions, min(len(questions), 1), replace=False))
    return np.random.choice(all_questions, min(len(all_questions), 10), replace=False).tolist()

# Example usage
random_questions = get_random_open_questions()
for question in random_questions:
    print(question)
