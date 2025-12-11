from typing import Dict, Any
import itertools
from GDesigner.prompt.prompt_set import PromptSet
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry

roles = itertools.cycle(['Science Expert',
                         'Scientific Analyst',
                         'Inspector',
                         'Researcher'])

ROLE_DESCRIPTION = {
    "Science Expert": 
        "You are a science expert. "
        "You will be given a scientific question and hints from other agents. "
        "Give your own reasoning process step by step based on hints. "
        "The last line of your output contains only the final answer as a letter, for example: The answer is C\n"
        "You will be given some examples you may refer to.",
    "Scientific Analyst":
        "You are a scientific analyst. "
        "You will be given a scientific question and analysis from other agents. "
        "You need to first analyze the problem step by step. "
        "Then provide your conclusion. "
        "The last line of your output contains only the final answer as a letter, for example: The answer is C\n"
        "You will be given some examples you may refer to.",
    "Inspector":
        "You are an Inspector. "
        "You will be given a scientific question and analysis from other agents. "
        "Check whether the logic and reasoning process is correct. "
        "Give your own reasoning process step by step based on hints. "
        "The last line of your output contains only the final answer as a letter, for example: The answer is C\n"
        "You will be given some examples you may refer to.",
    "Researcher":
        "You are a researcher. "
        "You will be given a scientific question and analysis from other agents. "
        "Provide detailed scientific reasoning. "
        "The last line of your output contains only the final answer as a letter, for example: The answer is C\n"
        "You will be given some examples you may refer to.",
}

ROLE_CONNECTION = [
    ('Scientific Analyst', 'Science Expert'),
    ('Scientific Analyst', 'Inspector'),
    ('Science Expert', 'Inspector'),
    ('Inspector', 'Science Expert'),
    ('Inspector', 'Researcher'),
    ('Researcher', 'Inspector'),
]

FEW_SHOT_DATA = {
    "Science Expert": "",
    "Scientific Analyst": "",
    "Inspector": "",
    "Researcher": "",
}

@PromptSetRegistry.register('gpqa')
class GPQAPromptSet(PromptSet):

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
    def get_answer_prompt(question, role="Science Expert"):
        return f"\n\nQ:{question}"

    @staticmethod
    def get_decision_constraint():
        return (
        "You will be given a scientific question and analysis from other agents. "
        "Please find the most reliable answer based on the analysis and results of other agents. "
        "Give reasons for making decisions. "
        "The last line of your output contains only the final answer as a letter, for example: The answer is C")
    
    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker. Good at analyzing and summarizing scientific problems, judging and summarizing other people's solutions, and giving final answers."
    
    @staticmethod
    def get_decision_few_shot():
        return ""
    
    @staticmethod
    def get_adversarial_answer_prompt(question):
        return f"\n\nQ:{question}"
    
    @staticmethod
    def get_query_prompt(question):
        return f"\n\nQ:{question}"
    
    @staticmethod
    def get_file_analysis_prompt(query, file):
        return f"Query: {query}\n\nFile: {file}"
    
    @staticmethod
    def get_websearch_prompt(question, query):
        return f"Question: {question}\n\nQuery: {query}"
    
    @staticmethod
    def get_distill_websearch_prompt(question, query, results):
        return f"Question: {question}\n\nQuery: {query}\n\nResults: {results}"
    
    @staticmethod
    def get_reflect_prompt(question, answer):
        return f"Question: {question}\n\nAnswer: {answer}"
    
    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return str(materials)
