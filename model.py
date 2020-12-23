import tensorflow as tf 

def recurrent_model(
    class_count,
    vocab_size,
    embedding_dim,
    max_sequence_len,
    latent_dim = 100
):
    #choose recurrent cell type and CRF instantiation
    cell_type = tf.keras.layers.LSTM
    

    #build sequential model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length= max_sequence_len),
            tf.keras.layers.Bidirectional(
                cell_type(
                    units=latent_dim,
                    return_sequences=True,
                    dropout=0.5, 
                    recurrent_dropout=0.5, 
                    kernel_initializer=tf.keras.initializers.he_normal()
                ),
            ),
            cell_type(
                units=latent_dim * 2, 
                return_sequences=True, 
                dropout=0.5, 
                recurrent_dropout=0.5, 
                kernel_initializer=tf.keras.initializers.he_normal()
            ),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    class_count+1, activation="softmax"
                )
            )
        ]
    )

    #compile model
    model.compile(
        loss = tf.keras.losses.CategoricalCrossentropy(),
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001, 
            beta_1=0.9, beta_2=0.999, epsilon=1e-07, 
            amsgrad=False,
            name='Adam'
        ),
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ]
    )
    return model
    