import pickle


class BaseModel:
    def __init__(self):
        """Initialize the BaseModel instance."""
        self.params, self.grads = None, None
        self.file_name = self.__class__.__name__ + ".pkl"

    def forward(self, *args):
        """Run the forward pass."""
        raise NotImplementedError

    def backward(self, *args):
        """Run the backward pass."""
        raise NotImplementedError

    def save_params(self, file_name=None):
        """Save model parameters to disk."""
        if file_name is None:
            file_name = self.file_name
        with open(file_name, "wb") as f:
            pickle.dump(self.params, f)

    def load_params(self, file_name=None):
        """Load model parameters from disk."""
        if file_name is None:
            file_name = self.file_name
        with open(file_name, "rb") as f:
            self.params = pickle.load(f)
