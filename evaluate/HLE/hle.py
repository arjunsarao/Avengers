import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

from datasets import Dataset, disable_progress_bars
from loguru import logger
from tqdm import tqdm

from core.inference import GeneratorFactory, GeneratorOutput
from core.routing import BaseRouter
from evaluate.base_evaluator import BaseEvaluator
from evaluate.deepscaler_rm import extract_answer, grade_answer_mathd, grade_answer_sympy

disable_progress_bars()

DATA_DIR = "data/HLE"

PROMPT = """{question}

Your response should be in the following format:

Explanation: {{your explanation for your answer choice}}
Answer: {{your chosen answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

MULTIPLE_CHOICE_ANSWER_PATTERN = r"(?i)Answer\s*:\s*([A-Z])\b"
EXACT_ANSWER_PATTERN = r"(?is)Answer\s*:\s*(.*?)(?:\n\s*Confidence\s*:|\Z)"


class HLEEvaluator(BaseEvaluator):
    def __init__(self, max_workers: int = 8, mode: str = "test"):
        super().__init__(max_workers=max_workers, mode=mode)
        self.task = "HLE"
        self.seed = 42

    def load_data(self, split: str):
        with open(os.path.join(DATA_DIR, "hle_physics.json"), "r") as f:
            data = json.load(f)

        data = Dataset.from_list(data)
        data = data.map(lambda x: self.format_prompt(x))
        if self.mode == "test":
            logger.warning(f"Using full local split for {self.task}; no calibration split is defined.")
        return data

    def format_prompt(self, item: Dict) -> Dict:
        return {"prompt": PROMPT.format(question=item["query"])}

    def extract_raw_answer(self, raw_datas: list[str], answer_type: str) -> list[str]:
        if answer_type == "multipleChoice":
            return [
                self.extract_normal_answer(text=data, answer_pattern=MULTIPLE_CHOICE_ANSWER_PATTERN).upper()
                for data in raw_datas
            ]

        return [self.extract_exact_answer(data) for data in raw_datas]

    def extract_exact_answer(self, text: str) -> str:
        if text is None:
            return ""

        answer = self.extract_normal_answer(text=text, answer_pattern=EXACT_ANSWER_PATTERN)
        if answer:
            return self.clean_extracted_answer(answer)

        boxed_answer = extract_answer(passage=text)
        if boxed_answer:
            return self.clean_extracted_answer(boxed_answer)

        return ""

    def clean_extracted_answer(self, answer: str) -> str:
        answer = answer.strip()
        answer = re.sub(r"(?is)\n\s*Confidence\s*:.*$", "", answer).strip()
        answer = re.sub(r"(?is)^Final Answer\s*:?", "", answer).strip()
        return answer.strip("`'\" ")

    def normalize_answer(self, answer: str) -> str:
        answer = self.clean_extracted_answer(answer)
        if "\\boxed" in answer:
            boxed_answer = extract_answer(passage=answer)
            if boxed_answer:
                answer = boxed_answer

        answer = answer.strip()
        if answer.startswith("$") and answer.endswith("$"):
            answer = answer[1:-1].strip()
        if answer.startswith("\\(") and answer.endswith("\\)"):
            answer = answer[2:-2].strip()
        if answer.startswith("\\[") and answer.endswith("\\]"):
            answer = answer[2:-2].strip()

        answer = re.sub(r"\s+", " ", answer)
        answer = answer.replace("\\,", "").replace("\\;", "")
        return answer.strip().rstrip(".").lower()

    def is_correct(self, prediction: str, answer: str, answer_type: str) -> bool:
        if not prediction:
            return False

        if answer_type == "multipleChoice":
            return prediction.upper() == answer.upper()

        prediction_normalized = self.normalize_answer(prediction)
        answer_normalized = self.normalize_answer(answer)
        if prediction_normalized == answer_normalized:
            return True

        try:
            return grade_answer_mathd(prediction, answer) or grade_answer_sympy(prediction, answer)
        except Exception as exc:
            logger.warning(f"Math grading fallback failed: {exc}")
            return False

    def process_output(self, output: GeneratorOutput, answer_type: str):
        full_prediction = self.extract_raw_answer(raw_datas=output.raw_output, answer_type=answer_type)
        prediction = Counter(full_prediction).most_common(1)[0][0]
        prediction_stats = self.count_prediction_frequency(predictions=full_prediction)

        return prediction, full_prediction, prediction_stats

    def evaluate(self, index: int, data: dict, router: BaseRouter, generator_config: dict):
        answer = data["gt"]
        answer_type = data["answer_type"]

        router_result = router.route(question=data["prompt"])
        generator = GeneratorFactory.create_generator(
            experts=router_result, generator_config=generator_config
        )  # type: ignore

        if generator_config["type"] == "model_switch":
            output: tuple[GeneratorOutput, GeneratorOutput] = generator.generate(question=data["prompt"])
            first_output, final_output = output
            prediction, full_prediction, prediction_stats = self.process_output(
                output=first_output, answer_type=answer_type
            )
            consistency_rate = prediction_stats[prediction]["frequency"]
            if consistency_rate < generator.consistency_rate_threshold:
                prediction, full_prediction, prediction_stats = self.process_output(
                    output=final_output, answer_type=answer_type
                )
                output = final_output
            else:
                output = first_output
        elif generator_config["type"] == "fast_slow":
            output: GeneratorOutput = generator.generate(question=data["prompt"])
            prediction, full_prediction, prediction_stats = self.process_output(
                output=output, answer_type=answer_type
            )
            consistency_rate = prediction_stats[prediction]["frequency"]
            if consistency_rate < generator.consistency_rate_threshold:
                slow_output = generator.slow_generate(question=data["prompt"])
                prediction, full_prediction, prediction_stats = self.process_output(
                    output=slow_output, answer_type=answer_type
                )
                if prediction == "":
                    logger.warning("slow_output is empty, use fast_output to replace.")
                    prediction, full_prediction, prediction_stats = self.process_output(
                        output=output, answer_type=answer_type
                    )
                else:
                    output = slow_output
        else:
            output: GeneratorOutput = generator.generate(question=data["prompt"])
            prediction, full_prediction, prediction_stats = self.process_output(
                output=output, answer_type=answer_type
            )

        self.update_tokens(prompt_tokens=output.prompt_tokens, completion_tokens=output.completion_tokens)
        is_correct = self.is_correct(prediction=prediction, answer=answer, answer_type=answer_type)

        return dict(
            index=index,
            id=data["id"],
            query=data["prompt"],
            origin_query=data["query"],
            prediction=prediction,
            full_prediction=full_prediction,
            prediction_stats=prediction_stats,
            raw_output=output.raw_output,
            answer=answer,
            answer_type=answer_type,
            category=data["category"],
            is_correct=is_correct,
            model_name=generator.model,
        )

    def evaluate_loop(self, router: BaseRouter, generator_config: dict):
        start_time = time.time()
        data = self.load_data(split="test")

        counter = 0
        type_counts = defaultdict(int)
        type_correct = defaultdict(int)
        results = []
        pbar = tqdm(total=len(data), desc=f"Evaluating {self.task} ...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.evaluate, index=idx, data=d, router=router, generator_config=generator_config)
                for idx, d in enumerate(data)
            ]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                type_counts[result["answer_type"]] += 1
                if result["is_correct"]:
                    counter += 1
                    type_correct[result["answer_type"]] += 1
                pbar.update(1)
        pbar.close()

        model_counts = self.calculate_model_counts(results=results)

        acc = counter / len(data)
        by_answer_type = {
            answer_type: type_correct[answer_type] / count
            for answer_type, count in type_counts.items()
        }
        end_time = time.time()
        logger.info(f"Task: {self.task}")
        logger.info(f"Accuracy: {acc}")
        logger.info(f"Accuracy by answer type: {by_answer_type}")
        logger.info(f"Time taken: {end_time - start_time} seconds")
        logger.info(f"Prompt tokens: {self.prompt_tokens}")
        logger.info(f"Completion tokens: {self.completion_tokens}")

        return {
            "performance": {
                "accuracy": acc,
                "by_answer_type": by_answer_type,
            },
            "time_taken": end_time - start_time,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model_counts": model_counts,
            "records": results,
        }
