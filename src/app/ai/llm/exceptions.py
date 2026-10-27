class LLMException(Exception):
    pass


class InvalidLLMProviderException(LLMException):
    pass


class InvalidRequest(LLMException):
    pass


class RateLimited(LLMException):
    pass


class LLMTimeout(LLMException):
    pass


class ProviderUnavailable(LLMException):
    pass
