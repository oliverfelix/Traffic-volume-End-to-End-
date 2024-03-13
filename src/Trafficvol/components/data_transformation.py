import os
from Trafficvol import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from Trafficvol.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.x = None
        self.y = None


    
    def data_preprocessing(self):
        
        data = pd.read_csv(self.config.data_path)
        
        
        data.drop(columns = ["holiday",'weather_description'],axis = 1, inplace = True) 
        
        data = data[data["weather_main"].isin(["Clouds", "Clear", "Mist", "Rain", "Snow", "Drizzle", "Haze", "Thunderstorm", "Fog"])]
        
        
        data['date_time'] = pd.to_datetime(data['date_time'], format='%Y-%m-%d %H:%M:%S')

        # Extract Date Features
        data['year'] = data['date_time'].dt.year
        data['month'] = data['date_time'].dt.month
        data['day'] = data['date_time'].dt.day
        data['day_of_week'] = data['date_time'].dt.dayofweek
        data['day_of_year'] = data['date_time'].dt.dayofyear

        # Extract Time Features
        data['hour'] = data['date_time'].dt.hour
        data['minute'] = data['date_time'].dt.minute


        # Create a new DataFrame for the modified 'date_time' column
        new_data = data.copy()

        # Drop the 'date_time' column from the original DataFrame
        data.drop(columns=['date_time'], inplace=True)
        
        
        data = pd.get_dummies(data, columns=['weather_main'], prefix='', prefix_sep='')
        
        
        self.x = data.drop("traffic_volume",axis = 1)
        self.y = data["traffic_volume"]


    def train_test_spliting(self):
        # Assuming data_preprocessing has already been called
        if self.x is None or self.y is None:
            raise ValueError("Data must be preprocessed before splitting.")

        # Split the data into training and test sets. (0.75, 0.25) split.
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.25, random_state=42)

        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Split data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)