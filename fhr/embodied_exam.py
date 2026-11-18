from oral_exam.exam import ExamManager


import logging
import random
from oral_exam.utils.helpers import EvaluationType, QuestionType


logger = logging.getLogger(__name__)

class EmbodiedExamManager(ExamManager):
    
    # removes audio output from ExamManager
    def speak(self, text):
        pass
    
    # sends the question to the embodyment platform
    def ask_question(self, question):
        pass
    
    def get_evaluation(self, question, student_answer):
        pass