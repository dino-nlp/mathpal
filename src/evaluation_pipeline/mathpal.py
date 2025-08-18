import pprint
import opik
from config import settings
from core import logger_utils
from core.opik_utils import add_to_dataset_with_sampling
from opik import opik_context

logger = logger_utils.get_logger(__name__)

class MathPal:
    def __init__(self):
        self.client = opik.Opik()

    def create_dataset(self, name: str, description: str, items: list[dict]) -> opik.Dataset:
        dataset = self.client.get_or_create_dataset(name=name, description=description)
        dataset.insert(items)
        return dataset