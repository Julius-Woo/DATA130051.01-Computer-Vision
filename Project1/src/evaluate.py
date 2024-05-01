import numpy as np


class Evaluator:
    '''Class to evaluate the model using new dataset.'''
    def __init__(self, model):
        self.model = model

    def eval(self, X_val, y_val, epochs=1, batch_size=None):
        steps = 1
        if batch_size is not None:
            steps = len(X_val) // batch_size
            if steps * batch_size < len(X_val):
                steps += 1

        self.model.loss.reset_cum_loss()
        self.model.accuracy.reset_cum()

        for step in range(steps):
            if batch_size is None:
                batch_X_val = X_val
                batch_y_val = y_val
            else:
                batch_X_val = X_val[step*batch_size:(step+1)*batch_size]
                batch_y_val = y_val[step*batch_size:(step+1)*batch_size]

            # Forward pass
            output = self.model.forward(batch_X_val, training=True)
            self.model.loss.calculate(output, batch_y_val)  # Test set does not include regularization loss
            predictions = self.model.output_layer.predictions(output)
            self.model.accuracy.calculate(predictions, batch_y_val)
        
        val_loss = self.model.loss.calc_cum_loss()
        val_accuracy = self.model.accuracy.calc_cum()
        
        print(f'Test---' +
              f'acc: {val_accuracy:.4f}, ' +
              f'loss: {val_loss:.4f}')
