

import tensorflow as tf

def get_actor_model(
        input_shape, output_shape, output_bounds, hidden=[256,256],
        activation='relu', activation_out='tanh', init_min=-0.003, 
        init_max=0.003,
    ):

    last_init = tf.random_uniform_initializer(minval=init_min, maxval=init_max)

    inputs = tf.keras.layers.Input(shape=(input_shape,))

    for idx, n in enumerate(hidden):
        layer = tf.keras.layers.Dense(
            n, activation=activation
        )
        if idx == 0:
            out = layer(inputs)
        else:
            out = layer(out)

    if hidden == []:
        out = inputs

    outputs = tf.keras.layers.Dense(
        output_shape, activation=activation_out, 
        kernel_initializer=last_init
    )(out)

    outputs = outputs * output_bounds
    model = tf.keras.Model(inputs, outputs)

    return model

def get_critic_model(
        input_shape_state, input_shape_action, hidden_state=[16,32], 
        hidden_action=[32], hidden_common=[256,256], activation='relu',
    ):

    state_input = tf.keras.layers.Input(shape=(input_shape_state))

    for idx, n in enumerate(hidden_state):
        layer = tf.keras.layers.Dense(
            n, activation=activation, kernel_regularizer='L2'
        )
        if idx == 0:
            state_out = layer(state_input)
        else:
            state_out = layer(state_out)
    
    if hidden_state == []:
        state_out = state_input

    # Action as input
    action_input = tf.keras.layers.Input(shape=(input_shape_action))

    for idx, n in enumerate(hidden_action):
        layer = tf.keras.layers.Dense(
            n, activation=activation, kernel_regularizer='L2'
        )
        if idx == 0:
            action_out = layer(action_input)
        else:
            action_out = layer(action_out)

    if hidden_action == []:
        action_out = action_input

    # Both are passed through seperate layer before concatenating
    concat = tf.keras.layers.Concatenate()([state_out, action_out])

    for idx, n in enumerate(hidden_common):
        layer = tf.keras.layers.Dense(
            n, activation=activation, kernel_regularizer='L2'
        )
        if idx == 0:
            common_out = layer(concat)
        else:
            common_out = layer(common_out)


    outputs = tf.keras.layers.Dense(1)(common_out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model
    