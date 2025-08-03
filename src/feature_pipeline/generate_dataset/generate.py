import sys
import json
import logging
from pathlib import Path

# To mimic using multiple Python modules, such as 'core' and 'feature_pipeline',
# we will add the './src' directory to the PYTHONPATH. This is not intended for
# production use cases but for development and educational purposes.
ROOT_DIR = str(Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)


from core import get_logger
from core.config import settings

logger = get_logger(__name__)

settings.patch_localhost()
logger.warning(
    "Patched settings to work with 'localhost' URLs. \
    Remove the 'settings.patch_localhost()' call from above when deploying or running inside Docker."
)

from comet_ml import Artifact, start
from core.db.qdrant import QdrantDatabaseConnector
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage
from huggingface_hub import login
from sklearn.model_selection import train_test_split
from .file_handler import FileHandler

client = QdrantDatabaseConnector()


class DatasetGenerator:
    def __init__(self, file_handler: FileHandler) -> None:
        self.file_handler = file_handler
    
    def generate_training_data(
        self, collection_name: str, data_type: str, grade_name: str
    ) -> None:
        assert (
            settings.COMET_API_KEY
        ), "COMET_API_KEY must be set in settings, fill it in your .env file."
        assert (
            settings.COMET_WORKSPACE
        ), "COMET_WORKSPACE must be set in settings, fill it in your .env file."
        
        assert (
            settings.COMET_PROJECT
        ), "COMET_PROJECT must be set in settings, fill it in your .env file."
        
        assert (
            settings.HUGGINGFACE_ACCESS_TOKEN
        ), "HUGGINGFACE_ACCESS_TOKEN must be set in settings, fill it in your .env file."
        
        cleaned_documents = self.fetch_all_cleaned_content(collection_name)
        num_cleaned_documents = len(cleaned_documents)
        logger.info(
            "Fetched all cleaned exams.",
            num_cleaned_documents=num_cleaned_documents
        )
        generated_instruct_dataset = []
        for document in cleaned_documents:
            parts = document.split('#### SOLUTION:')
            question_text = parts[0].replace('### QUESTION:', '').strip()
            solution_text = parts[1].strip()
            if len(question_text) < 20 or len(solution_text)==0:
                continue
            result_dict = {
                "question": question_text,
                "solution": solution_text
            }
            generated_instruct_dataset.append(result_dict)
        
        train_test_split = self._split_dataset(generated_instruct_dataset)
        self.push_to_comet(train_test_split, data_type, collection_name, grade_name)
        self.push_to_huggingface(train_test_split, data_type, grade_name)
    
    def _split_dataset(
        self, generated_instruct_dataset: list[dict], test_size: float = 0.1
    ) -> tuple[list[dict], list[dict]]:
        """Split dataset into train and test sets.

        Args:
            generated_instruct_dataset (dict): Dataset containing content and instruction pairs

        Returns:
            tuple[dict, dict]: Train and test splits of the dataset
        """

        if len(generated_instruct_dataset) == 0:
            return [], []

        train_data, test_data = train_test_split(
            generated_instruct_dataset, test_size=test_size, random_state=42
        )

        return train_data, test_data

    def push_to_comet(
        self,
        train_test_split: tuple[list[dict], list[dict]],
        data_type: str,
        collection_name: str, 
        grade_name: str,
        output_dir: Path = Path("generated_dataset"),
    ) -> None:
        output_dir.mkdir(exist_ok=True)

        try:
            logger.info(f"Starting to push data to Comet: {collection_name}")

            experiment = start(project_name=settings.COMET_PROJECT)

            training_data, testing_data = train_test_split

            file_name_training_data = output_dir / f"{collection_name}_training.json"
            file_name_testing_data = output_dir / f"{collection_name}_testing.json"

            logging.info(f"Writing training data to file: {file_name_training_data}")
            with file_name_training_data.open("w") as f:
                json.dump(training_data, f)

            logging.info(f"Writing testing data to file: {file_name_testing_data}")
            with file_name_testing_data.open("w") as f:
                json.dump(testing_data, f)

            logger.info("Data written to file successfully")

            artifact = Artifact(f"{data_type}-{grade_name}-instruct-dataset")
            artifact.add(file_name_training_data)
            artifact.add(file_name_testing_data)
            logger.info(f"Artifact created.")

            experiment.log_artifact(artifact)
            experiment.end()
            logger.info("Artifact pushed to Comet successfully.")

        except Exception:
            logger.exception(
                f"Failed to create Comet artifact and push it to Comet.",
            )

    def push_to_huggingface(
        self, 
        train_test_split: tuple[list[dict], list[dict]], 
        data_type: str,
        grade_name: str,):
        
        # Login to huggingface
        login(token=settings.HUGGINGFACE_ACCESS_TOKEN)
        logger.info("✅ Đã đăng nhập Hugging Face thành công")
        
        # create dataset
        features = Features({
            'question': Value('string'),
            'solution': Value('string')
        })
        train_data, test_data = train_test_split
        train_dataset = Dataset.from_list(train_data, features=features)
        test_dataset = Dataset.from_list(test_data, features=features)
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
        
        # Upload to huggingface
        repo_id=f"ngohongthai/{data_type}-{grade_name}-instruct-dataset"
        print(f"🚀 Đang upload dataset lên Hugging Face: {repo_id}")
        
        try:
            # Upload dataset
            dataset_dict.push_to_hub(
                repo_id=repo_id,
                private=False,  # Đặt True nếu muốn dataset private
                commit_message="Initial dataset upload"
            )
            
            print(f"✅ Đã upload dataset thành công!")
            print(f"🔗 Dataset URL: https://huggingface.co/datasets/{repo_id}")
            
        except Exception as e:
            print(f"❌ Lỗi khi upload dataset: {e}")
        
    def fetch_all_cleaned_content(self, collection_name: str) -> list:
        all_cleaned_contents = []

        scroll_response = client.scroll(collection_name=collection_name, limit=10000)
        points = scroll_response[0]

        for point in points:
            cleaned_content = point.payload["cleaned_content"]
            if cleaned_content:
                all_cleaned_contents.append(cleaned_content)

        return all_cleaned_contents
    
if __name__ == "__main__":
    file_handler = FileHandler()
    dataset_generator = DatasetGenerator(file_handler)
    
    collection_name = "cleaned_exams"
    data_type = "exam"
    grade_name = "sixth_grade"
    
    logger.info(
        "Generating training data.",
        collection_name=collection_name,
        data_type=data_type,
    )
    
    dataset_generator.generate_training_data(
        collection_name=collection_name, data_type=data_type, grade_name=grade_name
    )
