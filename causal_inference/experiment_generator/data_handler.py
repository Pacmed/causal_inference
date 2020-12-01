from data_warehouse_utils.dataloader import DataLoader

class DataHandler(DataLoader):

    def __init__(self):
        DataLoader.__init__(self)


    def get_causal_experiment(self):
        print('Hi!')