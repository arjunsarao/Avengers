from openai import OpenAI, NOT_GIVEN
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger

from core.experts.load_experts import Expert
from core.inference.base_generator import BaseGenerator, GeneratorOutput

class SelfConsistencyGenerator(BaseGenerator):
    def __init__(self, expert: Expert, generator_config: dict):
        self.client = expert.client
        self.model = expert.model_name
        self.config = generator_config
        self.samples = self.config.get("samples", 5)
        self.temperature = self.config.get("temperature", 0.2)
        self.top_p = self.config.get("top_p", 1.0)
        self.top_k = self.config.get("top_k", NOT_GIVEN)
    # 定义重试日志记录函数
    def _log_retry(retry_state):
        try:
            exception = retry_state.outcome.exception()
        except Exception:
            exception = None
        if exception:
            attempt = getattr(retry_state, "attempt_number", "?")
            logger.warning(
                f"Retrying SelfConsistencyGenerator.generate due to error: {type(exception).__name__}: {str(exception)}. Attempt {attempt}"
            )
        return None
    
    @retry(
        stop=stop_after_attempt(10),  # 最多重试10次
        wait=wait_exponential(multiplier=1, min=2, max=100),  # 指数退避策略：1*2^x 秒，最少2秒，最多100秒
        retry=retry_if_exception_type(Exception),  # 捕获所有异常进行重试
        before_sleep=_log_retry  # 重试前记录日志
    )
    def generate(self, question: str) -> GeneratorOutput:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": question}],
                temperature=self.temperature,
                top_p=self.top_p,
                n=self.samples,
                timeout=1_000,
            )
            choices = getattr(response, "choices", None)
            if choices is None and isinstance(response, dict):
                choices = response.get("choices")
            usage = getattr(response, "usage", None)

            if not choices:
                raise AttributeError(f"No choices in response: {repr(response)}")

            def _get_message_content(choice):
                try:
                    return choice.message.content
                except Exception:
                    try:
                        return choice["message"]["content"]
                    except Exception:
                        return None

            raw_output = [(_get_message_content(choice) or "") for choice in choices]
            assert len(raw_output) == self.samples, f"Expected {self.samples} samples, got {len(raw_output)}"
            first_output = _get_message_content(choices[0])
            if first_output is None:
                raise AttributeError("choices[0] has no message content")

            return GeneratorOutput(
                first_output=first_output,
                raw_output=raw_output,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            )
        except Exception as e:
            logger.exception(f"Error in SelfConsistencyGenerator.generate: {type(e).__name__}: {str(e)}")
            raise  # 重新抛出异常，让重试装饰器捕获