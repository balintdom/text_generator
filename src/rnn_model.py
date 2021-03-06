import tensorflow as tf
import time

class LSTMModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        """
            define an lstm language model, with embedding layer
            an LSTM layer and a fully-connected layer
        """
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        """
            forward step of the model, return states if return_states is True
        """
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.lstm.get_initial_state(x)
        x, *states = self.lstm(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x
        
class RNNGenerator(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        """
            define rnn based text generator
            
            Temperature: - is low the model will generate the most probable character
            if high the output will be more diverse
        """
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "" or "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(['', '[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
                        # Put a -inf at each bad index.
                        values=[-float('inf')]*len(skip_ids),
                        indices=skip_ids,
                        # Match the shape to the vocabulary
                        dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    def generate_text(self, starting_text, length):
        """
            generate length long text, starting with starting_text
        """
        start = time.time()
        states = None
        next_char = tf.constant([starting_text])
        result = [next_char]

        #generate new character as long as result is shorter than length
        for n in range(length):
            next_char, states = self.generate_one_step(next_char, states=states)
            result.append(next_char)

        result = tf.strings.join(result)
        end = time.time()

        return result[0].numpy().decode('utf-8')
        
    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                              return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        # Apply the prediction mask: prevent "" or "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states