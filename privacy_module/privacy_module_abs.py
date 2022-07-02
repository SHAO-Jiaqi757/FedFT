from abc import ABC, abstractmethod


class PrivacyModuleABC(ABC):
    @abstractmethod
    def privacy_mechanism(self) -> function:
        pass
    @abstractmethod
    def handle_response(self) -> function:
        pass