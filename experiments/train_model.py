def train_hred_model(model, optimizer, loss_func, loaders, args):
    for epoch_idx in range(args.n_epochs):
        print("Epoch {}:".format(epoch_idx))

        train_loss = 0.

        model.train()
        for batch_idx, batch in enumerate(loaders.train):
            optimizer.zero_grad()
            y_pred = model(batch['x_data'], batch['y_target'])

            out_size = y_pred.shape[-1]
            reshaped_target = batch['y_target'][1:].flatten()

            loss = loss_func(y_pred[1:].view(-1, out_size), reshaped_target)
            loss.backward()
            
            minibatch_loss = loss.item()
            print("minibatch_loss Loss: {:.3f}".format(minibatch_loss))

            train_loss += (minibatch_loss - train_loss) / (batch_idx + 1)

            optimizer.step()

        print("Training Loss: {:.3f}".format(train_loss))

        validation_loss = 0.
        model.eval()
        for batch_idx, batch in enumerate(loaders.val):
            y_pred = model(batch['x_data'], batch['y_target'], 0.)
            out_size = y_pred.shape[-1]
            reshaped_target = batch['y_target'][1:].flatten()
            loss = loss_func(y_pred[1:].view(-1, out_size), reshaped_target)

            minibatch_loss = loss.item()
            validation_loss += (minibatch_loss - validation_loss) / (batch_idx + 1)
        print("Validation Loss: {:.3f}".format(validation_loss))

    test_loss = 0.
    model.eval()
    for batch_idx, batch in enumerate(loaders.test):
        y_pred = model(batch['x_data'], batch['y_target'], 0.)
        out_size = y_pred.shape[-1]
        reshaped_target = batch['y_target'][1:].flatten()
        loss = loss_func(y_pred[1:].view(-1, out_size), reshaped_target)
        minibatch_loss = loss.item()
        test_loss += (minibatch_loss - test_loss) / (batch_idx + 1)
    print("Test Loss: {:.3f}".format(test_loss))
