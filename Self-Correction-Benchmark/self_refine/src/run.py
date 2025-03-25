import pandas as pd
from tqdm import tqdm


from task_init import GSMInit
from feedback import GSMFeedback

from utils import retry_parse_fail_prone_cmd
from customized_call import ChatWithOurServer
client = ChatWithOurServer(base_url = "http://0.0.0.0:65430/v1", model='Llama-3.1-8B-Instruct')
import sys
sys.path.append('/home/yueke/correct/hcr_selfcorrection/Self-Correction-Benchmark')
from utils.process_config import open_config
from model import create_model
from task import create_task
CODEX = "code-davinci-002"
# GPT3 = "text-davinci-003"
ENGINE = CODEX


@retry_parse_fail_prone_cmd
def iterative_gsm(question: str, max_attempts: int, feedback_type: str, temperature: float):

    # initialize all the required components

    # generation of the first fast version
    task_init = GSMInit(engine=ENGINE, prompt_examples="/home/yueke/correct/hcr_selfcorrection/Self-Correction-Benchmark/method/init.txt", temperature=temperature)

    # getting feedback
    if feedback_type == "naive":
        raise NotImplementedError
    else:
        task_feedback = GSMFeedback(engine=ENGINE, prompt_examples="/home/yueke/correct/hcr_selfcorrection/Self-Correction-Benchmark/method/feedback.txt", temperature=0.7)


    n_attempts = 0

    log = []

    while n_attempts < max_attempts:

        if n_attempts == 0:
            solution = task_init(solution=question)

        fb_and_maybe_soln = task_feedback(solution=solution)
        

        log.append({"attempt": n_attempts, "solution_curr": solution, "solution_fixed": fb_and_maybe_soln["solution"], "feedback": fb_and_maybe_soln["feedback"]})

        if "it is correct" in fb_and_maybe_soln["feedback"].lower():
            break

        solution = fb_and_maybe_soln["solution"]

        n_attempts += 1

    return log


def fix_gsm(gsm_task_file: str, question, max_attempts: int, outfile: str, feedback_type: str, temperature: float):


    slow_programs_df = pd.read_json(gsm_task_file, lines=True, orient="records")
    slow_programs_df["run_logs"] = None
    results = []
    for i, row in tqdm(slow_programs_df.iterrows(), total=len(slow_programs_df)):
        row_copy = row.to_dict()
        try:
            run_logs = iterative_gsm(question=row["input"], max_attempts=max_attempts, feedback_type=feedback_type, temperature=temperature)
            row_copy["run_logs"] = run_logs
            row_copy["generated_answer_ours"] = run_logs[-1]["solution_fixed"]
            row_copy["generated_answer_direct"] = run_logs[0]["solution_curr"]
            results.append(row_copy)
            if i % 10 == 0:
                pd.DataFrame(results).to_json(outfile + f".{i}.jsonl", orient="records", lines=True)
        except Exception as e:
            # raise e
            pass
    pd.DataFrame(results).to_json(outfile, orient="records", lines=True)
    return results




'''A test function for the class of SELF-REFINE'''
def test():
    import sys
    sys.path.append('/home/yueke/correct/hcr_selfcorrection/Self-Correction-Benchmark')
    from utils.process_config import open_config
    from model import create_model
    from task import create_task
    from tqdm import tqdm
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--gsm_task_file", type=str, default="F:/LAIR/self-refine/data/tasks/gsm/gsm.jsonl")
    parser.add_argument("--max_attempts", type=int, default=4)
    parser.add_argument("--outfile", type=str, default="F:/LAIR/self-refine/data/tasks/gsm/gsm_outputs.jsonl")
    parser.add_argument("--feedback_type", type=str, default="rich")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument('--model_config', type=str, default='/home/yueke/correct/hcr_selfcorrection/Self-Correction-Benchmark/config/model_config/llama_config.json')
    parser.add_argument('--task_config', type=str, default='/home/yueke/correct/hcr_selfcorrection/Self-Correction-Benchmark/config/task_config/gsm.json')
    parser.add_argument('--method', type=str, default='self_refine')
    
    args = parser.parse_args()
    model_config = open_config(config_path=args.model_config)
    model = create_model(model_config)

    task_config = open_config(config_path=args.task_config)
    task = create_task(task_config)
    data = task.get_data()
    
    args.outfile = f"{args.outfile}.fb_{args.feedback_type}.temp_{args.temperature}.engine_{ENGINE}.jsonl"
    logs = []
    for q, a in tqdm(zip(data['question'], data['final_answer'])):
        logs.append(fix_gsm(model, question=q, max_attempts=args.max_attempts, outfile=args.outfile, feedback_type=args.feedback_type, temperature=args.temperature))
        break
    for i, log in enumerate(logs):
        print(log["generated_answer_ours"])
        print(log["generated_answer_direct"])
def test():
    import json

    
    with open("/tmp/debug_gsm.jsonl", "w") as fout:
        fout.write(json.dumps({"input": "Twenty dozen cups cost $1200 less than the total cost of half a dozen plates sold at $6000 each. Calculate the total cost of buying each cup."}))
        
    logs = fix_gsm(
        gsm_task_file="/tmp/debug_gsm.jsonl", max_attempts=3, outfile="/tmp/test.jsonl", feedback_type="rich", temperature=0.0
    )
    for i, log in enumerate(logs):
        print(log["generated_answer_ours"])
        print(log["generated_answer_direct"])        
    
if "__main__" == __name__:
    test()
# if __name__ == "__main__":
#     import sys

#     if sys.argv[1] == "test":
#         test()
#     else:
#         import argparse
#         # args = argparse.ArgumentParser()
#         parser = argparse.ArgumentParser()
#         parser.add_argument('--model_config', type=str, default='/home/yueke/correct/hcr_selfcorrection/Self-Correction-Benchmark/config/model_config/llama_config.json')
#         parser.add_argument('--task_config', type=str, default='/home/yueke/correct/hcr_selfcorrection/Self-Correction-Benchmark/config/task_config/gsm.json')
#         parser.add_argument('--method', type=str, default='rci')
#         # args = parser.parse_args()
#         parser.add_argument("--gsm_task_file", type=str, default="data/tasks/gsm/gsm.jsonl")
#         parser.add_argument("--max_attempts", type=int, default=4)
#         parser.add_argument("--outfile", type=str, default="data/tasks/gsm/gsm_outputs.jsonl")
#         parser.add_argument("--feedback_type", type=str, default="rich")
#         parser.add_argument("--temperature", type=float, default=0.0)
#         args = parser.parse_args()
#         model_config = open_config(config_path=args.model_config)
#         model = create_model(model_config)
#         task_config = open_config(config_path=args.task_config)
#         task = create_task(task_config)
#         data = task.get_data()
#         args.outfile = f"{args.outfile}.fb_{args.feedback_type}.temp_{args.temperature}.engine_{ENGINE}.jsonl"
#         fix_gsm(gsm_task_file=args.gsm_task_file, max_attempts=args.max_attempts, outfile=args.outfile, feedback_type=args.feedback_type, temperature=args.temperature)