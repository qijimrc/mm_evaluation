import re
from typing import Dict
import string
import json
from nltk.tokenize import sent_tokenize
from Levenshtein import distance
import pandas as pd

from sat.helpers import print_rank0
from mmbench.common.registry import Registry
from mmbench.tasks.base_task import BaseTask
from mmbench.utils.chat_llama import LlamaChatter

@Registry.register_task('MathVista')
class MathVistaTask(BaseTask):
    def __init__(self, task_cfg, **kw_args):
        self.task_name = 'MathVista'
        super().__init__(task_cfg, **kw_args)

        self.chatter = LlamaChatter()


    def create_test_prompt(self, query, response):
        demo_prompt = DEMO_PROMPT.strip()
        test_prompt = f"Hint: {query}\n\n{response}"
        full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
        return full_prompt


    def extract_answer(self, response, problem, quick_extract=False):
        question_type = problem['question_type']
        answer_type = problem['answer_type']
        choices = problem['choices']
        query = problem['question']
        pid = problem['question_id']

        if response == "":
            return ""
        
        if question_type == 'multi_choice' and response in choices:
            return response
        
        if answer_type == "integer":
            try:
                extraction = int(response)
                return str(extraction)
            except:
                pass

        if answer_type == "float":
            try:
                extraction = str(float(response))
                return extraction
            except:
                pass

        # quick extraction
        if quick_extract:
            print("Quickly extracting answer...")
            # The answer is "text". -> "text"
            try:
                result = re.search(r'The answer is "(.*)"\.', response)
                if result:
                    extraction = result.group(1)
                    return extraction
            except:
                pass

        # general extraction
        try:
            full_prompt = self.create_test_prompt(query, response)
            extraction = self.chatter.chat(full_prompt)
            return extraction
        except Exception as e:
            print(e)
            print(f"Error in extracting answer for {pid}")

        return ""

    def get_most_similar(self, prediction, choices):
        """
        Use the Levenshtein distance (or edit distance) to determine which of the choices is most similar to the given prediction
        """
        distances = [distance(prediction, choice) for choice in choices]
        ind = distances.index(min(distances))
        return choices[ind]
        # return min(choices, key=lambda choice: distance(prediction, choice))


    def normalize_extracted_answer(self, extraction, choices, question_type, answer_type, precision):
        """
        Normalize the extracted answer to match the answer type
        """
        if question_type == 'multi_choice':
            # make sure the extraction is a string
            if isinstance(extraction, str):
                extraction = extraction.strip()
            else:
                try:
                    extraction = str(extraction)
                except:
                    extraction = ""
        
            # extract "A" from "(A) text"
            letter = re.findall(r'\(([a-zA-Z])\)', extraction)
            if len(letter) > 0:
                extraction = letter[0].upper()
            
            options = [chr(ord('A') + i) for i in range(len(choices))]
                
            if extraction in options:
                # convert option letter to text, e.g. "A" -> "text"
                ind = options.index(extraction)
                extraction = choices[ind]
            else:
                # select the most similar option
                extraction = self.get_most_similar(extraction, choices)
            assert extraction in choices

        elif answer_type == 'integer':
            try:
                extraction = str(int(float(extraction)))
            except:
                extraction = None

        elif answer_type == 'float':
            try:
                extraction = str(round(float(extraction), precision))
            except:
                extraction = None
            
        elif answer_type == 'list':
            try:
                extraction = str(extraction)
            except:
                extraction = None

        return extraction
        

    def safe_equal(self, prediction, answer):
        """
        Check if the prediction is equal to the answer, even if they are of different types
        """
        try:
            if prediction == answer:
                return True
            return False
        except Exception as e:
            print(e)
            return False


    def get_acc_with_contion(self, res_pd, key, value):
        if key == 'skills':
            # if value in res_pd[key]:
            total_pd = res_pd[res_pd[key].apply(lambda x: value in x)]
        else:
            total_pd = res_pd[res_pd[key] == value]

        correct_pd = total_pd[total_pd['true_false'] == True]
        acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100)
        return len(correct_pd), len(total_pd), acc
    
    def calc_scores(self, args, result_df) -> Dict:
        ## [1] Evaluate if the prediction is true or false
        results = {}
        full_pids = []
        for i, problem in result_df.iterrows():
            # Extact answer from prediction by GPT4 or other LLMs, referring to https://github.com/lupantech/MathVista/blob/main/evaluation/extract_answer.py
            # We use Llama-3-b8-instruct
            extraction = self.extract_answer(problem['preds'], problem)
            problem['extraction'] = extraction

            # Calculate scores, reffering to https://github.com/lupantech/MathVista/blob/main/evaluation/calculate_score.py
            pid = problem['question_id']
            answer = problem["answer"]
            full_pids.append(pid)

            choices = problem['choices']
            question_type = problem['question_type']
            answer_type = problem['answer_type']
            precision = problem['precision']

            # normalize the extracted answer to match the answer type
            prediction = self.normalize_extracted_answer(extraction, choices, question_type, answer_type, precision)

            # verify the prediction is true or false
            true_false = self.safe_equal(prediction, answer)
            
            problem['prediction'] = prediction
            problem['true_false'] = true_false

            results[pid] = problem
            results[pid]['query'] = problem['question']
            results[pid]['answer'] = problem['answer']
            results[pid]['prediction'] = prediction
            results[pid]['true_false'] = true_false


        ## [2] Calculate the average accuracy
        total = len(full_pids)
        correct = 0
        for pid in full_pids:
            if results[pid]['true_false']:
                correct += 1
        accuracy = str(round(correct / total * 100, 2))
        print(f"\nCorrect: {correct}, Total: {total}, Accuracy: {accuracy}%")

        scores = {"average": {"accuracy": accuracy, "correct": correct, "total": total}}
        print(scores, file=open('/share/home/qiji/repos/scores.json', 'w'))
        
        ## [3] Calculate the fine-grained accuracy scores

        # # merge the 'metadata' attribute into the data
        # for pid in results:
        #     results[pid].update(results[pid].pop('metadata'))

        # convert the data to a pandas DataFrame
        df = pd.DataFrame(results).T

        print(len(df))
        print("Number of test problems:", len(df))
        # assert len(df) == 1000 # Important!!!

        # asign the target keys for evaluation
        target_keys = ['question_type', 'answer_type', 'language', 'source', 'category', 'task', 'context', 'grade', 'skills']
        
        spec_scores = {}
        for key in target_keys:
            print(f"\nType: [{key}]")
            # get the unique values of the key
            if key == 'skills':
                # the value is a list
                values = []
                for i in range(len(df)):
                    values += df[key][i]
                values = list(set(values))
            else:
                values = df[key].unique()
            #print(values)

            # calculate the accuracy for each value
            spec_scores[key] = {}
            for value in values:
                correct, total, acc = self.get_acc_with_contion(df, key, value)
                if total > 0:
                    print(f"[{value}]: {acc}% ({correct}/{total})")
                    spec_scores[key][value] = {"accuracy": acc, "correct": correct, "total": total}
            
            # sort the scores by accuracy
            spec_scores[key] = dict(sorted(spec_scores[key].items(), key=lambda item: float(item[1]['accuracy']), reverse=True))
        print(spec_scores, file=open('/share/home/qiji/repos/spec_scores.json', 'w'))
        # save the scores

        metrics = {f"{tp}_{t}": t_vs for tp,tp_vs in scores.items() for t,t_vs in tp_vs.items()}
        metrics.update({f"{tp}_{t}_{met}": met_v for tp,tp_vs in spec_scores.items() for t,t_vs in tp_vs.items() for met,met_v in t_vs.items()})
        return metrics



DEMO_PROMPT = """
    Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

    Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
    Question: Which number is missing?

    Model response: The number missing in the sequence is 14.

    Extracted answer: 14

    Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
    Question: What is the fraction of females facing the camera?

    Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.

    Extracted answer: 0.6

    Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
    Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

    Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

    Extracted answer: 1.45

    Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
    Question: Between which two years does the line  graph saw its maximum peak?

    Model response: The line graph saw its maximum peak between 2007 and 2008.

    Extracted answer: [2007, 2008]

    Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
    Question: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5

    Model response: The correct answer is (B) 8/11.

    Extracted answer: B
    """
