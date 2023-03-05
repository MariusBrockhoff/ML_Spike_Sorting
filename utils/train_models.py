# -*- coding: utf-8 -*-
   
def train_model(model, save_weights):
    if model == "AutoPerceiver":
        lr = config.LEARNING_RATE
        wd = config.WEIGHT_DECAY
        
        if config.WITH_WD:
            optimizer = tfa.optimizers.AdamW(weight_decay=wd, learning_rate=lr)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        
        #subdir = datetime.datetime.now().strftime('run%Y%m%dT%H%M')
        #chk_dir = config.CHK_DIR + './checkpoints/'+subdir+'/' 
        
        loss_lst = []
        test_loss_lst = []
        
        
        for epoch in range(config.NUM_EPOCHS):
                
            if WITH_WARMUP:
                lr = config.LEARNING_RATE*min(epoch,config.LR_WARMUP)/config.LR_WARMUP
                if epoch>config.LR_WARMUP:
                    lr = config.LEARNING_RATE
                    lr = config.LR_FINAL + .5*(lr-config.LR_FINAL)*(1+np.cos(epoch*np.pi/config.NUM_EPOCHS))
                    wd = wd + .5*(wd-config.WD_FINAL)*(1+np.cos(epoch*np.pi/config.NUM_EPOCHS))
            else:
                lr = config.LR_FINAL + .5*(lr-config.LR_FINAL)*(1+np.cos(epoch*np.pi/config.NUM_EPOCHS))
                wd = wd + .5*(wd-config.WD_FINAL)*(1+np.cos(epoch*np.pi/config.NUM_EPOCHS))
        
            
            optimizer.learning_rate = lr
            optimizer.weight_decay = wd
            
            for step,batch in enumerate(dataset):      
                batch_s = batch
                with tf.GradientTape() as tape:                   
                  [ENC_state, logits, output] = autoencoder(batch_s)
                  mse = tf.keras.losses.MeanSquaredError()
                  loss = mse(batch_s, output)
                  grads = tape.gradient(loss,autoencoder.trainable_weights)
                  optimizer.apply_gradients(zip(grads,autoencoder.trainable_weights))
            loss_lst.append(loss)     
            
            
            #test loss
            for step,batch in enumerate(dataset_test):
              batch_t = batch 
              [ENC_state, logits, output] = autoencoder(batch_t)
              break
            test_loss= mse(batch_t, output)
            test_loss_lst.append(test_loss)
            
            print("Epoch: ", epoch+1, ", Train loss: ", loss, ", Test loss: ", test_loss)
       #if save_weights:
            #autoencoder.trainable_weights, 
            
       #implement model loader and saver 
        return loss_lst, test_loss_lst