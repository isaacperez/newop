import tensorflow as tf

import capa 


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Module functions: BaseModel
# -----------------------------------------------------------------------------
class BaseModel(tf.keras.Model):
    
    def __init__(self):
        super(BaseModel, self).__init__(name='BaseModel')

        self.num_factors = 6

        # ----------------------------------------------------------------------
        # Model definition
        # ----------------------------------------------------------------------
        self.factorLayer = capa.Factor(num_factors=self.num_factors, kernel_size=3)

    @tf.function
    def call(self, inputs, training=tf.constant(False)):
        return self.factorLayer(inputs)


