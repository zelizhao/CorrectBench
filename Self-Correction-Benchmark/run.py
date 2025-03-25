method = ["rci", "CAI", "Self-Refine", "CoVe", "Reflexion",
          "CRITIC", "Self-Debug", "Reflexion", "FLARE", "RARR",
          "SelFee", "DCOT", "REFINER", "RL4F", "Volcano"]


task = ["GSM8K","MATH", 
        "AQuA", "CSQA", "HotpotQA", "CommonSenseQA",
        "HumanEval"]



'''
Intrinsic self-correction test
'''
# 1.rci
## GSM8K
python evaluation_benchmark.py -model_config=GPT3.5 -task_config=GSM8K -method=rci
python evaluation_benchmark.py -model_config=LLaMA3.1_70B -task_config=GSM8K -method=rci

## MATH
python evaluation_benchmark.py -model_config=GPT3.5 -task_config=MATH -method=rci
python evaluation_benchmark.py -model_config=LLaMA3.1_70B -task_config=MATH -method=rci




'''
External self-correction test
'''
AQuA
# 1.RATT
## GSM8K
python evaluation_benchmark.py -model_config=GPT3.5 -task_config=GSM8K -method=ratt
python evaluation_benchmark.py -model_config=LLaMA3.1_70B -task_config=GSM8K -method=ratt

## MATH
python evaluation_benchmark.py -model_config=GPT3.5 -task_config=MATH -method=ratt
python evaluation_benchmark.py -model_config=LLaMA3.1_70B -task_config=MATH -method=ratt

## AQUA
python evaluation_benchmark.py -model_config=GPT3.5 -task_config=AQUA -method=ratt
python evaluation_benchmark.py -model_config=LLaMA3.1_70B -task_config=AQUA -method=ratt
