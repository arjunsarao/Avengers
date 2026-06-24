import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

from datasets import Dataset, disable_progress_bars
from loguru import logger
from tqdm import tqdm

from core.inference import GeneratorFactory, GeneratorOutput
from core.routing import BaseRouter
from evaluate.base_evaluator import BaseEvaluator
from evaluate.deepscaler_rm import extract_answer, grade_answer_mathd, grade_answer_sympy

disable_progress_bars()

DATA_DIR = "data/PHYSICS"

PROMPT = """Solve the following physics problem. The last line of your response should contain only your final answer in the format Answer: $ANSWER.

{question}
""".strip()

ANSWER_PATTERN = r"(?is)Answer\s*:\s*(.*?)(?:\n\s*(?:Confidence|Explanation)\s*:|\Z)"


class PhysicsEvaluator(BaseEvaluator):
    def __init__(self, max_workers: int = 8, mode: str = "test"):
        super().__init__(max_workers=max_workers, mode=mode)
        self.task = "PHYSICS"
        self.seed = 42

    def load_data(self, split: str):
        data = []
        data_dir = Path(DATA_DIR)
        for path in sorted(data_dir.glob("*.jsonl")):
            category = self.category_from_path(path)
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    item["category"] = category
                    item["source_file"] = str(path)
                    item["source_line"] = line_number
                    data.append(item)

        if not data:
            raise FileNotFoundError(f"No PHYSICS .jsonl files found in {DATA_DIR}")

        dataset = Dataset.from_list(data)
        dataset = dataset.map(lambda x: self.format_prompt(x))

        if self.mode == "test":
            logger.warning(f"Split data into train and test for {self.task}")
            split_data = dataset.train_test_split(test_size=0.3, seed=self.seed)
            train_data = split_data["train"]
            dataset = split_data["test"]
            logger.info(f"Calibration data: {len(train_data)}")
            logger.info(f"Test data: {len(dataset)}")
        else:
            logger.warning(f"Using full local split for {self.task}.")

        return dataset

    @staticmethod
    def category_from_path(path: Path) -> str:
        suffix = "_dataset_textonly.jsonl"
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
        return path.stem

    def format_prompt(self, item: Dict) -> Dict:
        return {"prompt": PROMPT.format(question=item["questions"])}

    def extract_raw_answer(self, raw_datas: list[str]) -> list[str]:
        return [self.extract_exact_answer(data) for data in raw_datas]

    def extract_exact_answer(self, text: str) -> str:
        if text is None:
            return ""

        answer = self.extract_normal_answer(text=text, answer_pattern=ANSWER_PATTERN)
        if answer:
            return self.clean_answer(answer)

        boxed_answer = extract_answer(passage=text)
        if boxed_answer:
            return self.clean_answer(boxed_answer)

        return ""

    def clean_answer(self, answer: str) -> str:
        answer = answer.strip()
        answer = re.sub(r"(?is)\n\s*(?:Confidence|Explanation)\s*:.*$", "", answer).strip()
        answer = re.sub(r"(?is)^Final Answer\s*:?", "", answer).strip()
        return answer.strip("`'\" ")

    def normalize_answer(self, answer: str) -> str:
        answer = self.clean_answer(answer)
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

    def is_correct(self, prediction: str, final_answers: list[str]) -> bool:
        if not prediction or not final_answers:
            return False

        prediction_normalized = self.normalize_answer(prediction)
        for answer in final_answers:
            answer_normalized = self.normalize_answer(answer)
            if prediction_normalized == answer_normalized:
                return True
            try:
                if grade_answer_mathd(prediction, answer) or grade_answer_sympy(prediction, answer):
                    return True
            except Exception as exc:
                logger.debug(f"Physics math grading fallback failed: {exc}")

        return False

    def process_output(self, output: GeneratorOutput):
        full_prediction = self.extract_raw_answer(raw_datas=output.raw_output)
        prediction = Counter(full_prediction).most_common(1)[0][0]
        prediction_stats = self.count_prediction_frequency(predictions=full_prediction)

        return prediction, full_prediction, prediction_stats

    def evaluate(self, index: int, data: dict, router: BaseRouter, generator_config: dict):
        final_answers = data.get("final_answers") or []

        router_result = router.route(question=data["prompt"])
        generator = GeneratorFactory.create_generator(
            experts=router_result, generator_config=generator_config
        )  # type: ignore

        if generator_config["type"] == "model_switch":
            output: tuple[GeneratorOutput, GeneratorOutput] = generator.generate(question=data["prompt"])
            first_output, final_output = output
            prediction, full_prediction, prediction_stats = self.process_output(output=first_output)
            consistency_rate = prediction_stats[prediction]["frequency"]
            if consistency_rate < generator.consistency_rate_threshold:
                prediction, full_prediction, prediction_stats = self.process_output(output=final_output)
                output = final_output
            else:
                output = first_output
        elif generator_config["type"] == "fast_slow":
            output: GeneratorOutput = generator.generate(question=data["prompt"])
            prediction, full_prediction, prediction_stats = self.process_output(output=output)
            consistency_rate = prediction_stats[prediction]["frequency"]
            if consistency_rate < generator.consistency_rate_threshold:
                slow_output = generator.slow_generate(question=data["prompt"])
                prediction, full_prediction, prediction_stats = self.process_output(output=slow_output)
                if prediction == "":
                    logger.warning("slow_output is empty, use fast_output to replace.")
                    prediction, full_prediction, prediction_stats = self.process_output(output=output)
                else:
                    output = slow_output
        else:
            output: GeneratorOutput = generator.generate(question=data["prompt"])
            prediction, full_prediction, prediction_stats = self.process_output(output=output)

        self.update_tokens(prompt_tokens=output.prompt_tokens, completion_tokens=output.completion_tokens)
        is_correct = self.is_correct(prediction=prediction, final_answers=final_answers)

        return dict(
            index=index,
            id=data["id"],
            query=data["prompt"],
            origin_query=data["questions"],
            prediction=prediction,
            full_prediction=full_prediction,
            prediction_stats=prediction_stats,
            raw_output=output.raw_output,
            answer=final_answers,
            solutions=data.get("solutions"),
            category=data["category"],
            source_file=data["source_file"],
            source_line=data["source_line"],
            is_correct=is_correct,
            model_name=generator.model,
        )

    def evaluate_loop(self, router: BaseRouter, generator_config: dict):
        start_time = time.time()
        data = self.load_data(split="test")

        counter = 0
        category_counts = defaultdict(int)
        category_correct = defaultdict(int)
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
                category_counts[result["category"]] += 1
                if result["is_correct"]:
                    counter += 1
                    category_correct[result["category"]] += 1
                pbar.update(1)
        pbar.close()

        model_counts = self.calculate_model_counts(results=results)

        acc = counter / len(data)
        by_category = {
            category: category_correct[category] / count
            for category, count in category_counts.items()
        }
        end_time = time.time()
        logger.info(f"Task: {self.task}")
        logger.info(f"Accuracy: {acc}")
        logger.info(f"Accuracy by category: {by_category}")
        logger.info(f"Time taken: {end_time - start_time} seconds")
        logger.info(f"Prompt tokens: {self.prompt_tokens}")
        logger.info(f"Completion tokens: {self.completion_tokens}")

        return {
            "performance": {
                "accuracy": acc,
                "by_category": by_category,
            },
            "time_taken": end_time - start_time,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model_counts": model_counts,
            "records": results,
        }
