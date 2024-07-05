import torch
import matplotlib.pyplot as plt

def add_salt_and_pepper_noise(images, salt_prob, pepper_prob):
    device = images.device
    batch_size, channels, height, width = images.shape
    
    # Generate random matrix for salt noise
    salt_noise = torch.rand(batch_size, channels, height, width).to(device)
    salt_mask = salt_noise < salt_prob
    
    # Generate random matrix for pepper noise
    pepper_noise = torch.rand(batch_size, channels, height, width).to(device)
    pepper_mask = pepper_noise < pepper_prob
    
    # Apply salt and pepper noise
    noisy_images = images.clone()
    noisy_images[salt_mask] = 1.0  # Set to max value (salt)
    noisy_images[pepper_mask] = 0.0  # Set to min value (pepper)
    
    return noisy_images

def train_generator(gen_model, train_loader, test_loader, num_epochs, gen_criterion, gen_optimizer, batch_size, device):
    G_loss = []
    G_loss_val = []
    for epoch in range(num_epochs):
        gen_model.train()
        avg_loss = 0
        for i, (img, c) in enumerate(train_loader, 0):
            image = img.to(device)
            image_noise = add_salt_and_pepper_noise(image, 0.8, 0.8)
            gen_optimizer.zero_grad()

            output = gen_model(image_noise)
            loss = gen_criterion(image, output)
            loss.backward()
            gen_optimizer.step()
            
            avg_loss += loss.item()

        avg_loss /= len(train_loader) / batch_size
        G_loss.append(avg_loss)
        print("Epoch = {}, Avg loss = {}".format(epoch, avg_loss))

        gen_model.eval()
        with torch.no_grad():
            avg_loss = 0
            for i, (img, c) in enumerate(test_loader, 0):
                image = img.to(device)
                noise = torch.randn(*image.shape, device=device)
                output = gen_model(noise)
                loss = gen_criterion(image, output)
                
                avg_loss += loss.item()
                
            avg_loss /= len(test_loader) / batch_size
            G_loss_val.append(avg_loss)
            print("Epoch = {}, Avg loss = {}".format(epoch, avg_loss))
    return {"G_loss" :G_loss, "G_loss_val": G_loss_val}
        

def train_discriminator(disc_model, combined_train_loader, combined_test_loader, num_epochs, disc_criterion, disc_optimizer, batch_size, device):
    D_loss_train = []
    D_loss_val = []
    train_acc = []
    val_acc = []

    for epoch in range(num_epochs):
        disc_model.train()
        avg_loss = 0.0
        train_acc_epoch = 0.0
        for i, (img, c) in enumerate(combined_train_loader, 0):
            disc_optimizer.zero_grad()
            
            image = img.to(device)
            c = c.to(device).float()

            output = disc_model(image).squeeze()
            loss = disc_criterion(output, c)

            loss.backward()
            disc_optimizer.step()
            
            avg_loss += loss.item()

            train_acc_epoch += torch.sum((output.round() == c).float()) / batch_size
        train_acc_epoch /= len(combined_train_loader)
    
        avg_loss /= len(combined_train_loader) / batch_size
        print("Epoch = {}, Train Avg loss = {}".format(epoch, avg_loss))
        print("Train Acc = {}".format(train_acc_epoch))
        train_acc.append(train_acc_epoch)
        D_loss_train.append(avg_loss)

        disc_model.eval()
        with torch.no_grad():
            val_acc_epoch = 0
            avg_loss = 0
            for i, (img, c) in enumerate(combined_test_loader, 0):
                image = img.to(device)
                c = c.to(device).float()

                output = disc_model(image).squeeze()
                loss = disc_criterion(output, c)
                
                avg_loss += loss.item()

                val_acc_epoch += torch.sum((output.round() == c).float()) / batch_size
            val_acc_epoch /= len(combined_test_loader)

            avg_loss /= len(combined_test_loader) / batch_size
            print("Epoch = {}, Val Avg loss = {}".format(epoch, avg_loss))
            print("Val Acc = {}".format(val_acc_epoch))
            D_loss_val.append(avg_loss)
            val_acc.append(val_acc_epoch)
    return {"train_acc": train_acc, "val_acc" : val_acc, "train_loss": D_loss_train, "val_loss": D_loss_val}