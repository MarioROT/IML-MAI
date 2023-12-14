import numpy as np
import pandas as pd

class InstanceSelection():
    def __init__(self,
                 features: pd.Dataframe,
                 labels: pd.series,
                 ):
        self.features = features
        self.labels = labels



