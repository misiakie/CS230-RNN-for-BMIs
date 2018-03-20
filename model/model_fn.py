"""Define the model."""

import tensorflow as tf


def build_model(mode, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (neurons_activity, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """
    spike_neurons  = inputs['spike_neurons']
    is_training  = (mode == 'train')
    if params.model_version == 'lstm':
        if is_training:
            keep_prob = params.keep_prob
        else:
            keep_prob = 1.0
        keep_prob = tf.placeholder_with_default(1.0,[])
        
        if params.is_bidirectionnal:
            fwd_LSTM_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
            fwd_dropout_cell = tf.nn.rnn_cell.DropoutWrapper(fwd_LSTM_cell, 
                                          input_keep_prob=keep_prob, output_keep_prob=keep_prob)
            
            bwd_LSTM_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
            bwd_dropout_cell = tf.nn.rnn_cell.DropoutWrapper(bwd_LSTM_cell, 
                                          input_keep_prob=keep_prob, output_keep_prob=keep_prob)

            outputs, outputs_states = tf.nn.bidirectional_dynamic_rnn(
                                        fwd_dropout_cell,
                                        bwd_dropout_cell,
                                        spike_neurons,
                                        dtype = tf.float32)
        
            state_fw, state_bw = outputs_states
            c_fw, h_fw = state_fw
            c_bw, h_bw = state_bw
            h = tf.concat([h_fw,h_bw], 1)
        
        else:
            fwd_LSTM_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
            fwd_dropout_cell = tf.nn.rnn_cell.DropoutWrapper(fwd_LSTM_cell, 
                                          input_keep_prob=keep_prob, output_keep_prob=keep_prob)
            try:
                if (params.num_layers>1):
                    fwd_dropout_cell = tf.contrib.rnn.MultiRNNCell([fwd_dropout_cell] * params.num_layers,
                                                                   state_is_tuple=True)
            except:
                pass
            output, output_state = tf.nn.dynamic_rnn( 
                                     fwd_dropout_cell,
                                     spike_neurons,
                                     dtype = tf.float32)
            c_fw, h = output_state
        
        if params.model_with_quadrants:
            n_outputs = 11
        else:
            n_outputs = params.num_classes

        logits = tf.layers.dense(h, n_outputs)


    else:
        raise NotImplementedError("Unknown model version: {}".format(params.model_version))

    return logits


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training  = (mode == 'train')
    numTarget    = inputs['numTarget']
    label        = inputs['label']
    isSuccessful = inputs['isSuccessful']
    delayTime    = inputs['delayTime']
    
    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(mode, inputs, params)

    # Define loss and accuracy depending on the model we want to use
    is_acceptable_try = tf.reshape(tf.logical_and(isSuccessful, (delayTime>=params.delay_time_min)), [-1])

    if params.model_with_quadrants:
        loss_circle = tf.reduce_mean(tf.boolean_mask(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=label[:,:3], logits=logits[:,:3]), is_acceptable_try))
        loss_quadrant = tf.reduce_mean(tf.boolean_mask(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=label[:,3:7], logits=logits[:,3:7]), is_acceptable_try))
        loss_angle = tf.reduce_mean(tf.boolean_mask(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=label[:,7:], logits=logits[:,7:]), is_acceptable_try))
        loss = loss_circle + loss_quadrant + loss_angle
    else:
        loss_per_example = tf.boolean_mask(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=numTarget, logits=logits), is_acceptable_try) 
        loss = tf.reduce_mean(loss_per_example)
    
    lambda_l2_reg = params.lambda_l2_reg
    l2_loss = lambda_l2_reg * tf.add_n(
        [tf.nn.l2_loss(tf_var)
            for tf_var in tf.trainable_variables(scope='model')
            if not ("noreg" in tf_var.name or "bias" in tf_var.name)])
            

    if params.model_with_quadrants:
        predicted_circle = tf.cast(tf.argmax(logits[:,:3], axis=1), tf.int32)
        predicted_quadrant = tf.cast(tf.argmax(logits[:,3:7], axis=1), tf.int32)
        predicted_angle = tf.cast(tf.argmax(logits[:,7:], axis=1), tf.int32)
    
        label_circle = tf.cast(tf.argmax(label[:,:3], axis=1), tf.int32)
        label_quadrant = tf.cast(tf.argmax(label[:,3:7], axis=1), tf.int32)
        label_angle = tf.cast(tf.argmax(label[:,7:], axis=1), tf.int32)
        
        is_right_circle   = tf.equal(predicted_circle, label_circle)
        is_right_quadrant = tf.equal(predicted_quadrant, label_quadrant)
        is_right_angle    = tf.equal(predicted_angle, label_angle)
        is_right_target   = tf.logical_and(is_right_angle,
                                           tf.logical_and(is_right_quadrant, is_right_circle))
        
        accuracy_circle   = tf.reduce_mean(tf.boolean_mask(tf.cast(is_right_circle, tf.float32),
                                            is_acceptable_try))
        accuracy_quadrant = tf.reduce_mean(tf.boolean_mask(tf.cast(is_right_quadrant, tf.float32),
                                            is_acceptable_try))
        accuracy_angle    = tf.reduce_mean(tf.boolean_mask(tf.cast(is_right_angle, tf.float32),
                                            is_acceptable_try))
        accuracy          = tf.reduce_mean(tf.boolean_mask(tf.cast(is_right_target, tf.float32),
                                            is_acceptable_try))
        
    else:
        predicted_target= tf.argmax(logits, axis=1)
        accuracy        = tf.reduce_mean(tf.cast(tf.boolean_mask(
                            tf.nn.in_top_k(logits, numTarget, 1 ), is_acceptable_try),
                            tf.float32))
    
    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        loss = loss + l2_loss
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        if params.model_with_quadrants:
            metrics = {
                'accuracy_circle': tf.metrics.accuracy(labels=label_circle, predictions=predicted_circle,
                                                weights=tf.cast(is_acceptable_try, tf.float32)),
                'accuracy_quadrant': tf.metrics.accuracy(labels=label_quadrant, predictions=predicted_quadrant,
                                                weights=tf.cast(is_acceptable_try, tf.float32)),
                'accuracy_angle': tf.metrics.accuracy(labels=label_angle, predictions=predicted_angle,
                                                weights=tf.cast(is_acceptable_try, tf.float32)),
                'accuracy': tf.metrics.mean(is_right_target, weights=tf.cast(is_acceptable_try, tf.float32)),                                         
                'loss': tf.metrics.mean(loss)
            }
            
        else:
            metrics = {
                'accuracy': tf.metrics.accuracy(labels=numTarget, predictions=predicted_target,
                                                weights=tf.cast(is_acceptable_try, tf.float32)),
                'accuracy_top_3': tf.metrics.mean(
                        tf.nn.in_top_k(predictions=logits, targets=numTarget, k=3),
                        weights=tf.cast(is_acceptable_try, tf.float32)),
                'accuracy_top_5': tf.metrics.mean(
                        tf.nn.in_top_k(predictions=logits, targets=numTarget, k=5),
                        weights=tf.cast(is_acceptable_try, tf.float32)),
                'accuracy_top_10': tf.metrics.mean(
                        tf.nn.in_top_k(predictions=logits, targets=numTarget, k=10),
                        weights=tf.cast(is_acceptable_try, tf.float32)),
                'loss': tf.metrics.mean(loss)
            }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    if params.model_with_quadrants:
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy_circle', accuracy_circle)
        tf.summary.scalar('accuracy_quadrant', accuracy_quadrant)
        tf.summary.scalar('accuracy_angle', accuracy_angle)
    else:
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec['loss']             = loss
    model_spec['accuracy']         = accuracy
    
    if params.model_with_quadrants:
        model_spec['accuracy_circle']  = accuracy_circle
        model_spec['accuracy_quadrant']= accuracy_quadrant
        model_spec['accuracy_angle']   = accuracy_angle
    else:
        model_spec["predictions"] = predicted_target
        
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
