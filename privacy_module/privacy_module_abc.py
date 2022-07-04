from abc import ABC, abstractmethod


class PrivacyModuleABC(ABC):
    @abstractmethod
    def privacy_mechanism(self) -> callable :
        pass
    @abstractmethod
    def handle_response(self) -> callable :
        pass