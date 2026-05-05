import pickle


class BaseModel:
    def __init__(self):
        self.params, self.grads = None, None

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    def save_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + ".pkl"
        with open(file_name, "wb") as f:
            pickle.dump(self.params, f)

    def load_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + ".pkl"
        with open(file_name, "rb") as f:
            self.params = pickle.load(f)
