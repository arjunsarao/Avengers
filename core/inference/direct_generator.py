from openai import OpenAI, NOT_GIVEN
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger

from core.experts.load_experts import Expert
from core.inference.base_generator import BaseGenerator, GeneratorOutput

class DirectGenerator(BaseGenerator):
    def __init__(self, expert: Expert, generator_config: dict):
        self.client = expert.client
        self.model = expert.model_name
        self.config = generator_config
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
                f"Retrying DirectGenerator.generate due to error: {type(exception).__name__}: {str(exception)}. Attempt {attempt}"
            )
        return None
    
    @retry(
        stop=stop_after_attempt(5),  # 最多重试5次
        wait=wait_exponential(multiplier=1, min=2, max=60),  # 指数退避策略：1*2^x 秒，最少2秒，最多60秒
        retry=retry_if_exception_type(Exception),  # 捕获所有异常进行重试
        before_sleep=_log_retry  # 重试前记录日志
    )
    def generate_with_retry(self, question: str) -> GeneratorOutput:
        if "Distill" in self.model or "EXAOME" in self.model:
            question += "Don't make your reasoning and thinking too long.\n"
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": question}],
                temperature=self.temperature,
                top_p=self.top_p,
                timeout=500,
            )

            # robustly extract choices and usage from different client response types
            choices = None
            usage = None
            try:
                if hasattr(response, "choices"):
                    choices = response.choices
                elif isinstance(response, dict):
                    choices = response.get("choices")
                else:
                    # fallback: try attribute access on nested objects
                    choices = getattr(response, "choices", None)
            except Exception:
                choices = None

            try:
                if hasattr(response, "usage"):
                    usage = response.usage
                elif isinstance(response, dict):
                    usage = response.get("usage")
                else:
                    usage = getattr(response, "usage", None)
            except Exception:
                usage = None

            if not choices:
                raise AttributeError(f"No choices found in response: {repr(response)}")

            # handle choice message content extraction for multiple response shapes
            def _get_message_content(choice):
                try:
                    return choice.message.content
                except Exception:
                    try:
                        # dict-like
                        return choice["message"]["content"]
                    except Exception:
                        try:
                            return choice["text"]
                        except Exception:
                            return None

            first = _get_message_content(choices[0])
            if first is None:
                raise AttributeError("choices[0] has no message content")

            raw = []
            for c in choices:
                content = _get_message_content(c)
                raw.append(content if content is not None else "")

            prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
            completion_tokens = getattr(usage, "completion_tokens", 0) if usage is not None else 0

            return GeneratorOutput(
                first_output=first,
                raw_output=raw,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as e:
            logger.exception(f"Error in DirectGenerator.generate: {type(e).__name__}: {str(e)}, model_name: {self.model}")
            raise  # 重新抛出异常，让重试装饰器捕获
    
    def generate(self, question: str) -> GeneratorOutput:
        try:
            return self.generate_with_retry(question=question)
        except Exception as e:
            logger.error(
                f"Error in DirectGenerator.generate after all retries: "
                f"{str(e)}, model_name: {self.model}"
            )
            return GeneratorOutput(
                first_output="failed to generate",
                raw_output=["failed to generate"],
                prompt_tokens=0,
                completion_tokens=0
            )
