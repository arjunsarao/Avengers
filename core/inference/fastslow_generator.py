from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from loguru import logger
from openai import NOT_GIVEN, OpenAI
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

from core.experts.load_experts import Expert
from core.inference.base_generator import BaseGenerator, GeneratorOutput


class FastSlowGenerator(BaseGenerator):
    def __init__(self, fast_expert: Expert, slow_expert: Expert, generator_config: dict):
        self.name = self.__class__.__name__
        # get clients and models from fast and slow experts
        self.fast_expert = fast_expert
        self.slow_expert = slow_expert
        self.fast_client = fast_expert.client
        self.fast_model = fast_expert.model_name
        self.slow_client = slow_expert.client
        self.slow_model = slow_expert.model_name
        self.model = [self.fast_model]
        
        # get fast and slow config
        self.config = generator_config
        self.fast_samples = self.config.get("fast_samples", 10)
        self.fast_temperature = self.config.get("fast_temperature", 0.7)
        self.fast_top_p = self.config.get("fast_top_p", 1.0)
        self.slow_samples = self.config.get("slow_samples", 1)
        self.slow_temperature = self.config.get("slow_temperature", 0.7)
        self.slow_top_p = self.config.get("slow_top_p", 1.0)
        self.consistency_rate_threshold = self.config.get("consistency_rate_threshold", 0.8)
        
        self.fast_max_retries = 20
        self.slow_max_retries = 3
        # define final results
        self.fast_results = None
        self.slow_results = None
        self.final_results = None
    
    def _generate_with_retry(self, client: OpenAI, model: str, question: str, mode: str = "fast") -> GeneratorOutput:
        max_retries = self.fast_max_retries if mode == "fast" else self.slow_max_retries
        
        def _log_retry(retry_state):
            try:
                exception = retry_state.outcome.exception()
            except Exception:
                exception = None
            if exception:
                attempt = getattr(retry_state, "attempt_number", "?")
                logger.warning(
                    f"Retrying FastSlowGenerator.generate due to error: {type(exception).__name__}: {str(exception)}. Attempt {attempt}"
                )
            return None
        
        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            retry=retry_if_exception_type(Exception),
            before_sleep=_log_retry
        )
        def _generate_impl():
            temperature = self.fast_temperature if mode == "fast" else self.slow_temperature
            top_p = self.fast_top_p if mode == "fast" else self.slow_top_p
            samples = self.fast_samples if mode == "fast" else self.slow_samples
            timeout = 2_000 if mode == "fast" else 200_000
            if mode == "slow":
                slow_prompt = "\nDon't make your reasoning and thinking too long.\n"
                question_with_prompt = question + slow_prompt
            else:
                question_with_prompt = question
                
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": question_with_prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    n=samples,
                    timeout=timeout,
                )

                # robust extraction
                choices = getattr(response, "choices", None) if response is not None else None
                if choices is None and isinstance(response, dict):
                    choices = response.get("choices")
                usage = getattr(response, "usage", None) if response is not None else None

                if not choices:
                    raise AttributeError(f"No choices in response: {repr(response)}")

                def _get_message_content(choice):
                    try:
                        return choice.message.content
                    except Exception:
                        try:
                            return choice["message"]["content"]
                        except Exception:
                            try:
                                return choice.get("text")
                            except Exception:
                                return None

                raw_output = [(_get_message_content(choice) or "") for choice in choices]
                assert len(raw_output) == samples, f"Mode={mode}, Expected {samples} samples, got {len(raw_output)}"

                first_output = _get_message_content(choices[0])
                if first_output is None:
                    raise AttributeError("choices[0] has no message content")

                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage is not None else 0

                return GeneratorOutput(
                    first_output=first_output,
                    raw_output=raw_output,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            except Exception as e:
                logger.exception(f"Error in FastSlowGenerator._generate: {type(e).__name__}: {str(e)}, error model: {model}, mode: {mode}")
                raise
                
        return _generate_impl()

    def generate(self, question: str) -> GeneratorOutput:
        self.fast_results = self._generate_with_retry(self.fast_client, self.fast_model, question, "fast")
        return self.fast_results
    
    def slow_generate(self, question: str) -> GeneratorOutput:
        try:
            self.slow_results = self._generate_with_retry(self.slow_client, self.slow_model, question, "slow")
            self.final_results = GeneratorOutput(
                first_output = self.slow_results.first_output,
                raw_output = self.slow_results.raw_output,
                prompt_tokens = self.slow_results.prompt_tokens + self.fast_results.prompt_tokens,
                completion_tokens = self.slow_results.completion_tokens + self.fast_results.completion_tokens
            )
            self.model = [self.slow_model]
            return self.final_results
        except Exception as e:
            logger.warning(f"Slow model generation failed after all retries: {str(e)}. Falling back to fast model results.")
            return self.fast_results