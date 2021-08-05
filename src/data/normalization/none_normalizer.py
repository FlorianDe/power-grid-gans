from src.data.normalization.base_normalizer import BaseNormalizer, T


class NoneNormalizer(BaseNormalizer):
    def fit(self, data: T) -> None:
        pass

    def is_fitted(self) -> bool:
        return True

    def normalize(self, data: T) -> T:
        return data

    def renormalize(self, data: T) -> T:
        return data
