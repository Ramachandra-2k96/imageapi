import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow_model_optimization.sparsity import keras as sparsity
model_path = 'enhanced_signature_verification_model3.keras'
model = load_model(model_path)
pruning_params = {
    'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                 final_sparsity=0.90,
                                                 begin_step=0,
                                                 end_step=1000,
                                                 frequency=100)
}
model_for_pruning = sparsity.prune_low_magnitude(model, **pruning_params)
model_for_pruning.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

