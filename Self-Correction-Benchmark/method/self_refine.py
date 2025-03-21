import pandas as pd
from tqdm import tqdm
from src.utils import Prompt
from together import Together
client = Together(api_key = '4373b3d6d0f7e3aaecea460eb1a0bb99d1f94349eaf13945937751e9d55a5002')

from src.utils import retry_parse_fail_prone_cmd
class GSMInit(Prompt):
    def __init__(self, prompt_examples: str, model: str, temperature: float) -> None:
        super().__init__(
            question_prefix="# Q: ",
            answer_prefix="# solution using Python:\n",
            intra_example_sep="\n",
            inter_example_sep="\n\n",
            model=model,
            temperature=temperature,
        )
        self.model = model
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, prompt_examples) -> str:
        with open(prompt_examples, "r") as f:
            self.prompt = f.read()
    
    def make_query(self, solution: str) -> str:
        solution = solution.strip()
        query = f"{self.prompt}{self.question_prefix}{solution}{self.intra_example_sep}{self.answer_prefix}"
        return query

    def __call__(self, solution: str) -> str:
        generation_query = self.make_query(solution)
        # messages = [{"role": "user", "content": generation_query}]
        # completion = client.chat.completions.create(
        #     messages=messages,
        #     model=self.model,
        #     max_tokens=300,
        #     stop=[self.inter_example_sep],
        #     temperature=self.temperature,
        # )
        solution_code = self.model.query(generation_query)

        # solution_code = completion.choices[0].message.content


        return solution_code.strip()
class GSMFeedback(Prompt):
    def __init__(self, model: str, prompt_examples: str, temperature: float, max_tokens: int = 600) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n### END ###\n\n",
            model = model,
            temperature = temperature
        )
        
        self.max_tokens = max_tokens
        self.instruction = "# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good." if "naive" not in prompt_examples else "# There is an error in the code above."
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, examples_path: str) -> str:
        with open(examples_path, "r") as f:
            self.prompt = f.read()
    
    def __call__(self, solution: str):
        generation_query = self.make_query(solution=solution)
        print(generation_query)
        # print(1/0)
        # messages = [{"role": "user", "content": generation_query}]
        # completion = client.chat.completions.create(
        #     messages=messages,
        #     model=self.model,
        #     max_tokens=self.max_tokens,
        #     stop=["### END"],
        #     temperature=self.temperature,
        # )
        entire_output = self.model.query(generation_query)
        # entire_output = completion.choices[0].message.content
        

        print(entire_output)
        if "### END" in entire_output:
            entire_output = entire_output.split("### END")[0]

        improved_soln = entire_output.split("def solution():")[1]
        feedback = entire_output.split("def solution():")[0]
        improved_soln = "def solution():" + improved_soln.rstrip()
        self.update_prompt(solution=solution, improved_soln=improved_soln, feedback=feedback)
        return {"solution": improved_soln, "feedback": feedback}

    def make_query(self, solution: str):
        
        solution = f"""{self.question_prefix}{solution}{self.intra_example_sep}{self.instruction}{self.answer_prefix}"""
        return f"{self.prompt}{solution}"
    
    
    def update_prompt(self, solution: str, improved_soln: str, feedback: str):
        prefix = f"""{self.question_prefix}{solution}{self.intra_example_sep}{self.instruction}{self.answer_prefix}"""
        
        gen_ans = f"""

{feedback}

{improved_soln.rstrip()}{self.inter_example_sep}"""

        new_example = f"{prefix}{gen_ans}"
        self.prompt = f"{self.prompt}{new_example}"
        
CODEX = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
# GPT3 = "text-davinci-003"
ENGINE = CODEX


@retry_parse_fail_prone_cmd
def iterative_gsm(model, question: str, max_attempts: int, feedback_type: str, temperature: float):

    # initialize all the required components

    # generation of the first fast version
    task_init = GSMInit(model=model, prompt_examples="F:/LAIR/self-refine/data/prompt/gsm/init.txt", temperature=temperature)

    # getting feedback
    if feedback_type == "naive":
        raise NotImplementedError
    else:
        task_feedback = GSMFeedback(model=model, prompt_examples="F:/LAIR/self-refine/data/prompt/gsm/feedback.txt", temperature=0.7)


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


def fix_gsm(model, question: str, max_attempts: int, outfile: str, feedback_type: str, temperature: float):


    slow_programs_df = pd.DataFrame([{"input": question}])
    slow_programs_df["run_logs"] = None
    results = []

    # 遍历每一行数据（这里仅有一行）
    for i, row in tqdm(slow_programs_df.iterrows(), total=len(slow_programs_df)):
        row_copy = row.to_dict()
        try:
            # 调用 iterative_gsm 函数处理每个问题
            run_logs = iterative_gsm(model, question=row["input"], max_attempts=max_attempts, feedback_type=feedback_type, temperature=temperature)
            
            # 将生成的反馈和修正答案添加到结果中
            row_copy["run_logs"] = run_logs
            row_copy["generated_answer_ours"] = run_logs[-1]["solution_fixed"]
            row_copy["generated_answer_direct"] = run_logs[0]["solution_curr"]
            results.append(row_copy)

            # 每处理 10 条记录就输出一次中间结果（虽然这里每次只有一行）
            if i % 10 == 0:
                pd.DataFrame(results).to_json(outfile + f".{i}.jsonl", orient="records", lines=True)

        except Exception as e:
            # 如果出现异常，跳过这条记录
            pass

    # 将所有处理结果保存到最终输出文件
    pd.DataFrame(results).to_json(outfile, orient="records", lines=True)

    # 返回所有结果
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
    parser.add_argument("--gsm_task_file", type=str, default="F:/LAIR/self-refine/data/tasks/gsm/gsm.jsonl")
    parser.add_argument("--max_attempts", type=int, default=4)
    parser.add_argument("--outfile", type=str, default="F:/LAIR/self-refine/data/tasks/gsm/gsm_outputs.jsonl")
    parser.add_argument("--feedback_type", type=str, default="rich")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument('--model_config', type=str, default='/home/yueke/correct/hcr_selfcorrection/Self-Correction-Benchmark/config/model_config/llama_config.json')
    parser.add_argument('--task_config', type=str, default='/home/yueke/correct/hcr_selfcorrection/Self-Correction-Benchmark/config/task_config/gsm.json')
    parser.add_argument('--method', type=str, default='relf_refine')
    
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
        
    
if "__main__" == __name__:
    test()
  
