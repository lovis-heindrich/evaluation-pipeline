import torch
from torch import nn, optim
import torch.nn.functional as F
from train import train
import utils

class Generator(nn.Module):
  def __init__(self, noise_dim=100, output_dim=3):
    super(Generator, self).__init__()
    self.conv1 = nn.ConvTranspose2d(noise_dim, 512, 4, 1, 0) # output 256x4x4
    self.bn1 = nn.BatchNorm2d(512)
    self.conv2 = nn.ConvTranspose2d(512, 256, 4, 1, 0) # output 256x4x4
    self.bn2 = nn.BatchNorm2d(256)
    self.conv3 = nn.ConvTranspose2d(256, 128, 4, 2, 0) # output 128x8x8
    self.bn3 = nn.BatchNorm2d(128)
    self.conv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1) # output 64x16x16
    self.bn4 = nn.BatchNorm2d(64)
    self.conv5 = nn.ConvTranspose2d(64, output_dim, 4, 2, 1) # output 3x32x32

  def forward(self, x):
    x = x.view(x.shape[0], x.shape[1], 1, 1)
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    x = F.relu(self.bn4(self.conv4(x)))
    x = torch.tanh(self.conv5(x))
    return x

class CGenerator(nn.Module):
  def __init__(self, noise_dim=100, output_dim=3, num_classes=14):
    super(CGenerator, self).__init__()
    self.embed0 = nn.Embedding(num_classes, 50)
    self.conv0 = nn.ConvTranspose2d(50, 1, 4, 1, 0)

    self.conv1 = nn.ConvTranspose2d(noise_dim, 512, 4, 1, 0) # output 256x4x4
    self.bn1 = nn.BatchNorm2d(512)
    self.conv2 = nn.ConvTranspose2d(513, 256, 4, 1, 0) # output 256x4x4
    self.bn2 = nn.BatchNorm2d(256)
    self.conv3 = nn.ConvTranspose2d(256, 128, 4, 2, 0) # output 128x8x8
    self.bn3 = nn.BatchNorm2d(128)
    self.conv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1) # output 64x16x16
    self.bn4 = nn.BatchNorm2d(64)
    self.conv5 = nn.ConvTranspose2d(64, output_dim, 4, 2, 1) # output 3x32x32

  def forward(self, x, label):
    # Class label
    label = self.embed0(label)
    label = label.view(label.shape[0], label.shape[1], 1, 1)
    label = self.conv0(label)
    # Noise input
    x = x.view(x.shape[0], x.shape[1], 1, 1)
    x = F.relu(self.bn1(self.conv1(x)))
    # Concat label
    x = torch.cat([x, label], dim=1)
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    x = F.relu(self.bn4(self.conv4(x)))
    x = torch.tanh(self.conv5(x))
    return x

class Discriminator(nn.Module):
  def __init__(self, input_dim=3, wgan=False):
    super(Discriminator, self).__init__()
    self.wgan = wgan
    self.conv1 = nn.Conv2d(input_dim, 64, 4, 2, 1) # output 64x16x16
    self.conv2 = nn.Conv2d(64, 128, 4, 2, 1) # output 128x8x8
    self.bn2 = nn.BatchNorm2d(128)
    self.conv3 = nn.Conv2d(128, 256, 4, 2, 1) # output 256x4x4
    self.bn3 = nn.BatchNorm2d(256)
    self.conv4 = nn.Conv2d(256, 512, 4, 2, 1) # output 512x4x4
    self.bn4 = nn.BatchNorm2d(512)
    self.conv5 = nn.Conv2d(512, 1, 4, 1, 0)

  def forward(self, x):
    x = F.leaky_relu(self.conv1(x), 0.02)
    x = F.leaky_relu(self.bn2(self.conv2(x)), 0.02)
    x = F.leaky_relu(self.bn3(self.conv3(x)), 0.02)
    x = F.leaky_relu(self.bn4(self.conv4(x)), 0.02)
    x = self.conv5(x)
    if self.wgan:
      return x
    return torch.sigmoid(x)

class CDiscriminator(nn.Module):
  def __init__(self, input_dim=3, num_classes=14, wgan=False):
    super(CDiscriminator, self).__init__()
    self.wgan = wgan

    self.embed0 = nn.Embedding(num_classes, 50)
    self.conv0 = nn.ConvTranspose2d(50, 1, 64, 1, 0) # output 1x64x64

    self.conv1 = nn.Conv2d(input_dim+1, 64, 4, 2, 1) # output 64x16x16
    self.conv2 = nn.Conv2d(64, 128, 4, 2, 1) # output 128x8x8
    self.bn2 = nn.BatchNorm2d(128)
    self.conv3 = nn.Conv2d(128, 256, 4, 2, 1) # output 256x4x4
    self.bn3 = nn.BatchNorm2d(256)
    self.conv4 = nn.Conv2d(256, 512, 4, 2, 1) # output 512x4x4
    self.bn4 = nn.BatchNorm2d(512)
    self.conv5 = nn.Conv2d(512, 1, 4, 1, 0)

  def forward(self, x, label):
    label = self.embed0(label)
    label = label.view(label.shape[0], label.shape[1], 1, 1)
    label = self.conv0(label)
    x = torch.cat([x, label], dim=1)

    x = F.leaky_relu(self.conv1(x), 0.02)
    x = F.leaky_relu(self.bn2(self.conv2(x)), 0.02)
    x = F.leaky_relu(self.bn3(self.conv3(x)), 0.02)
    x = F.leaky_relu(self.bn4(self.conv4(x)), 0.02)
    x = self.conv5(x)
    if self.wgan:
      return x
    return torch.sigmoid(x)

class Classifier(nn.Module):
  def __init__(self, input_dim=3, num_classes=10):
    super(Classifier, self).__init__()
    self.conv1 = nn.Conv2d(input_dim, 64, 4, 2, 1) # output 64x16x16
    self.conv2 = nn.Conv2d(64, 128, 4, 2, 1) # output 128x8x8
    self.bn2 = nn.BatchNorm2d(128)
    self.conv3 = nn.Conv2d(128, 256, 4, 2, 1) # output 256x4x4
    self.bn3 = nn.BatchNorm2d(256)
    self.conv4 = nn.Conv2d(256, num_classes, 8, 1, 0) # output 256x2x2
    self.dropout = nn.Dropout2d(0.6)

  def forward(self, x):
    x = self.dropout(F.relu(self.conv1(x)))
    x = self.dropout(F.relu(self.bn2(self.conv2(x))))
    x = self.dropout(F.relu(self.bn3(self.conv3(x))))
    x = self.conv4(x)
    return x.view(x.shape[0], x.shape[1])

def initGenerator(noise_size=100, num_channels=3, conditional=True, lee=True, wgan=False):
  if conditional:
    generator = CGenerator(noise_size, num_channels)
  else:
    generator = Generator(noise_size, num_channels)

  return generator

class gan_trainer():
  def __init__(self, num_classes, device, trainloader, testloader, image_size=64, noise_size=100, num_channels=3, beta=1, normalize_images=False):
    self.noise_size = noise_size
    self.num_channels = num_channels
    self.device = device
    self.loss = nn.BCELoss()
    self.c_loss = nn.CrossEntropyLoss()
    self.trainloader = trainloader
    self.testloader = testloader
    self.num_classes = num_classes
    self.image_size = image_size
    self.beta = beta
    self.normalize_images = normalize_images

  def initGAN(self, conditional=True, lee=True, wgan=False):
    if conditional:
        generator = CGenerator(self.noise_size, self.num_channels, num_classes=self.num_classes)
        discriminator = CDiscriminator(self.num_channels, wgan=wgan, num_classes=self.num_classes)
    else:
        generator = Generator(self.noise_size, self.num_channels)
        discriminator = Discriminator(self.num_channels, wgan=wgan)
        
    discriminator.to(self.device)
    generator.to(self.device)

    d_optimizer = optim.Adam(discriminator.parameters(), 0.0002, (0.5, 0.999))
    g_optimizer = optim.Adam(generator.parameters(), 0.0002, (0.5, 0.999))

    if lee:
        classifier = Classifier(self.num_channels, self.num_classes)
        classifier.to(self.device)
        c_optimizer = optim.Adam(classifier.parameters(), 0.0002)
        train(classifier, c_optimizer, self.c_loss, 1, self.trainloader, self.testloader, self.device, True)
        return generator, g_optimizer, discriminator, d_optimizer, classifier, c_optimizer
    
    return generator, g_optimizer, discriminator, d_optimizer, None, None

  def train_discriminator(self, discriminator, generator, d_optimizer, inputs, labels, batch_size, conditional=True, wgan=False):
    with torch.no_grad():
        if conditional:
            fake_examples = generator(torch.randn(batch_size,self.noise_size, 1, 1).to(self.device), labels)
        else:
            fake_examples = generator(torch.randn(batch_size,self.noise_size, 1, 1).to(self.device))
        fake_examples = torch.clamp(fake_examples, -1, 1)
    
    # WGAN requires clamped weights
    if wgan:
        for p in discriminator.parameters():
            p.data.clamp_(-0.1, 0.1)

    d_optimizer.zero_grad()


    # Discriminator fake examples
    if conditional:
        outputs = discriminator(fake_examples.view(batch_size, self.num_channels, self.image_size, self.image_size), labels)
    else: 
        outputs = discriminator(fake_examples.view(batch_size, self.num_channels, self.image_size, self.image_size))

    if wgan:
        fake_loss = torch.mean(outputs.view(-1))
    else:
        fake_loss = self.loss(outputs.view(-1), torch.zeros(batch_size).to(self.device) + (torch.rand(batch_size).to(self.device)*0.1))
    #fake_loss.backward()

    # Discriminator real examples
    if conditional:
        outputs = discriminator(inputs, labels)
    else:
        outputs = discriminator(inputs)
    if wgan:
        real_loss = -1*torch.mean(outputs.view(-1))
    else:
        real_loss = self.loss(outputs.view(-1), torch.ones(inputs.shape[0]).to(self.device) - (torch.rand(inputs.shape[0]).to(self.device)*0.1))
    #real_loss.backward()

    d_loss = real_loss + fake_loss
    d_loss.backward()
    # Discriminator update
    d_optimizer.step()
    return d_loss.detach().item()#fake_loss + real_loss

  def train_generator(self, discriminator, generator, g_optimizer, labels, batch_size, classifier=None, sricharan=False, uniform=None, conditional=True, lee=False, wgan=False):
      g_optimizer.zero_grad()
      # Standard Generator GAN loss
      if conditional:
          fake_examples = generator(torch.randn(batch_size, self.noise_size, 1, 1).to(self.device), labels)
          fake_examples = torch.clamp(fake_examples, -1, 1)
          outputs = discriminator(fake_examples.view(batch_size, self.num_channels, self.image_size, self.image_size), labels)
      else:
          fake_examples = generator(torch.randn(batch_size, self.noise_size, 1, 1).to(self.device))
          fake_examples = torch.clamp(fake_examples, -1, 1)
          outputs = discriminator(fake_examples.view(batch_size, self.num_channels, self.image_size, self.image_size))
      if wgan:
          gen_loss = -1*torch.mean(outputs.view(-1))
      else:
          gen_loss = self.loss(outputs.view(-1), torch.ones(batch_size).to(self.device) - (torch.rand(batch_size).to(self.device)*0.1))

      # Generator KL loss
      if lee:
          preds = classifier(fake_examples)
          p_model = F.log_softmax(preds, 1)
          kl_loss = F.kl_div(p_model, uniform) * self.num_classes
          total_gen_loss = gen_loss + self.beta * kl_loss
          if sricharan:
              total_gen_loss *= -1
      else:
          total_gen_loss = gen_loss
      
      total_gen_loss.backward()
      # Generator update
      g_optimizer.step()
      return total_gen_loss.detach().item(), fake_examples

  def train_classifier(self, classifier, generator, c_optimizer, inputs, labels, batch_size, uniform=None, conditional=True):
      if conditional:
          fake_examples = generator(torch.randn(batch_size, self.noise_size, 1, 1).to(self.device), labels)
      else:
          fake_examples = generator(torch.randn(batch_size,self.noise_size, 1, 1).to(self.device))
      fake_examples = torch.clamp(fake_examples, -1, 1)
      c_optimizer.zero_grad()

      # Classifier fake examples
      outputs = classifier(fake_examples.view(batch_size, self.num_channels, self.image_size, self.image_size))
      softmax_outputs = F.log_softmax(outputs, 1)
      kl_loss = F.kl_div(softmax_outputs, uniform)* self.num_classes

      # Classifier real examples
      outputs = F.log_softmax(classifier(inputs), 1)
      class_loss = F.nll_loss(outputs, labels)
      total_loss = class_loss + self.beta * kl_loss
      total_loss.backward()

      # Classifier update
      c_optimizer.step()
      return total_loss

  def generate_examples(self, generator, n, labels=None, conditional=False):
      with torch.no_grad():
          generator.eval()
          if conditional:
              x = generator(torch.randn(n, self.noise_size, 1, 1).to(self.device), labels[:n].to(self.device)).to("cpu")
          else:
              x = generator(torch.randn(n, self.noise_size, 1, 1).to(self.device)).to(self.device).to("cpu")
          return x

  def trainGAN(self, generator, g_optimizer, discriminator, d_optimizer, n, classifier=None, c_optimizer=None, baseline=None, use_baseline=False, sricharan=False, conditional=True, lee=False, discriminator_steps=5, wgan=False, early_stopping=1, classifier_stopping=0):
    d_running_losses = []
    g_running_losses = []
    if lee:
        c_running_losses = []
    if use_baseline:
        baseline_acc = []
        baseline_act = []

    for epoch in range(n):
        d_running_loss = 0.0
        g_running_loss = 0.0
        if lee:
            c_running_loss = 0.0
        n_batches = 0
        if use_baseline:
            total = 0.0
            correct = 0.0
            activations = []
        for i, data in enumerate(self.trainloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(self.device), data[1].to(self.device)

            if self.normalize_images:
              inputs = (inputs-0.5)*2
            
            batch_size = inputs.shape[0]
            uniform = torch.Tensor(batch_size, self.num_classes).fill_((1./self.num_classes)).to(self.device)

            # Train discriminator every n batches
            if i%discriminator_steps == 0:
                d_running_loss += self.train_discriminator(discriminator, generator, d_optimizer, inputs, labels, batch_size, conditional=conditional, wgan=wgan)

            # Train generator
            g_loss, fake_examples = self.train_generator(discriminator, generator, g_optimizer, labels, batch_size, classifier=classifier, uniform=uniform, conditional=conditional, sricharan=sricharan, lee=lee, wgan=wgan)
            g_running_loss += g_loss

            # Train classifier
            if lee:
                c_running_loss += self.train_classifier(classifier, generator, c_optimizer, inputs, labels, batch_size, uniform=uniform, conditional=conditional)
            
            # Check baseline accuracy on GAN
            if use_baseline:
                with torch.no_grad():
                    output, activation, predicted = baseline.get_prediction(fake_examples) 
                    correct += (predicted == labels).sum().item()
                    activations.append(activation)
                    total += labels.shape[0]
            # check for nan
            #if torch.isnan(fake_examples.view(-1)).sum().item() + torch.isnan(outputs.view(-1)).sum().item() > 0:
            #    print("NAN IN CLASSIFIER:", torch.isnan(outputs.view(-1)).sum().item())
            #    print("NAN IN GENERATOR:", torch.isnan(fake_examples.view(-1)).sum().item())
            #    break

            n_batches += 1
        print('[%d] discriminator loss: %.3f' %
                (epoch + 1, d_running_loss / n_batches))
        print('[%d] generator loss: %.3f' %
                (epoch + 1, g_running_loss / n_batches))
        d_running_losses.append(d_running_loss / n_batches)
        g_running_losses.append(g_running_loss / n_batches)
        if lee:
            print('[%d] classifier loss: %.3f' %
                    (epoch + 1, c_running_loss / n_batches))
            
            c_running_losses.append(c_running_loss / n_batches)
            if (c_running_loss / n_batches) < classifier_stopping:
                print("Stopping")
                break
        if use_baseline:
            print("Classifier accuracy on GAN:", correct/total, "Activation:", torch.mean(torch.cat(activations).to("cpu")).item())
            baseline_acc.append(correct/total)
            baseline_act.append(torch.mean(torch.cat(activations).to("cpu")).item())

            if correct/total > early_stopping:
                print("Stopping...")
                break
        utils.gridshow(inputs.to("cpu")[0:16,:,:,:], True)
        utils.gridshow(torch.clamp(fake_examples.view(batch_size, self.num_channels, self.image_size, self.image_size).detach().to("cpu"), -1, 1)[0:32,:,:,:], True)
    res = {"Discriminator loss": d_running_losses,
           "Generator loss": g_running_losses}
    if lee:
        res["Lee classifier loss"] = c_running_losses
    if use_baseline:
        res["Baseline accuracy"] = baseline_acc
        res["Baseline activation"] = baseline_act
    models = {
        "generator": generator,
        "discriminator": discriminator,
        "classifier": classifier
    }
    return res, models