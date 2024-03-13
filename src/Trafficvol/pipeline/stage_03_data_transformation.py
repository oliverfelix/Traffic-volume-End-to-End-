from Trafficvol.config.configuration import ConfigurationManager
from Trafficvol.components.data_transformation import DataTransformation
from Trafficvol import logger
from pathlib import Path


STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        try:
            
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            data_transformation.data_preprocessing()
            data_transformation.train_test_spliting()

        except FileNotFoundError as file_not_found_error:
            logger.error(f"File not found: {file_not_found_error}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise e
