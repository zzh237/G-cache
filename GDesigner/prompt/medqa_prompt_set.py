from typing import Dict, Any
import itertools
from GDesigner.prompt.prompt_set import PromptSet
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry

roles = itertools.cycle(['Medical Expert',
                         'Clinical Analyst',
                         'Inspector',
                         'Medical Researcher'])

ROLE_DESCRIPTION = {
    "Medical Expert": 
        "You are a medical expert. "
        "You will be given a medical question and hints from other agents. "
        "Give your own reasoning process step by step based on hints. "
        "The last line of your output contains only the final answer as a letter, for example: The answer is C\n"
        "You will be given some examples you may refer to.",
    "Clinical Analyst":
        "You are a clinical analyst. "
        "You will be given a medical question and analysis from other agents. "
        "You need to first analyze the problem step by step. "
        "Then provide your conclusion. "
        "The last line of your output contains only the final answer as a letter, for example: The answer is C\n"
        "You will be given some examples you may refer to.",
    "Inspector":
        "You are an Inspector. "
        "You will be given a medical question and analysis from other agents. "
        "Check whether the logic and reasoning process is correct. "
        "Give your own reasoning process step by step based on hints. "
        "The last line of your output contains only the final answer as a letter, for example: The answer is C\n"
        "You will be given some examples you may refer to.",
    "Medical Researcher":
        "You are a medical researcher. "
        "You will be given a medical question and analysis from other agents. "
        "Provide detailed medical reasoning. "
        "The last line of your output contains only the final answer as a letter, for example: The answer is C\n"
        "You will be given some examples you may refer to.",
}

ROLE_CONNECTION = [
    ('Clinical Analyst', 'Medical Expert'),
    ('Clinical Analyst', 'Inspector'),
    ('Medical Expert', 'Inspector'),
    ('Inspector', 'Medical Expert'),
    ('Inspector', 'Medical Researcher'),
    ('Medical Researcher', 'Inspector'),
]

FEW_SHOT_DATA = {
    "Medical Expert": "",
    "Clinical Analyst": "",
    "Inspector": "",
    "Medical Researcher": "",
}

@PromptSetRegistry.register('medqa')
class MedQAPromptSet(PromptSet):

    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_constraint(role):
        return ROLE_DESCRIPTION[role]

    def get_description(self, role):
        return ROLE_DESCRIPTION[role]
    
    def get_role_connection(self):
        return ROLE_CONNECTION
        
    @staticmethod
    def get_format():
        return "natural language"

    @staticmethod
    def get_answer_prompt(question, role="Medical Expert"):
        return f"\n\nQ:{question}"

    @staticmethod
    def get_decision_constraint():
        return (
        "You will be given a medical question and analysis from other agents. "
        "Please find the most reliable answer based on the analysis and results of other agents. "
        "Give reasons for making decisions. "
        "The last line of your output contains only the final answer as a letter, for example: The answer is C")
    
    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker. Good at analyzing and summarizing medical problems, judging and summarizing other people's solutions, and giving final answers."
    
    @staticmethod
    def get_decision_few_shot():
        return ""
