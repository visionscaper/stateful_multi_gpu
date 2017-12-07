# stateful_multi_gpu
Experimental utility in development to build stateful RNN models for multi GPU training.

## How to use stateful_multi_gpu()
See `example.py` for a toy example of how the utility function can be used.

To make stateful models in Keras you need to provide the batch size you are using.  
`stateful_multi_gpu()` ensures that the right batch size is used throughout the model.

To use this utility function you need to create a `inputs_generator` method that can make 
inputs for your model, for any batch size, and a `model_generator` method that can make
your model for any batch size.

For instance, as follows:

```python
def inputs_generator(batch_size):
    rnn_input = Input(
        name="rnn-input-%d" % batch_size,
        batch_shape=(batch_size, seq_len, num_symbols))

    return rnn_input
```

```python
def model_generator(batch_size):
    inputs = inputs_generator(batch_size)

    layer_output = RNNLayer(
        state_size,
        stateful=stateful_model,
        return_sequences=True)(inputs)
    outputs = TimeDistributed(Dense(num_classes))(layer_output)

    return Model(inputs=inputs, outputs=outputs)
```

`stateful_multi_gpu` uses these generator methods to 
create model inputs that expect batch size `batch_size` and model replica's (one per GPU) that expect 
batch size `batch_size` // `num_gpus`. E.g.:

```python
parallel_model = stateful_multi_gpu(inputs_generator, model_generator, training_batch_size, num_gpus)
```

It is important that `batch_size` is wholly dividable by `num_gpus`.