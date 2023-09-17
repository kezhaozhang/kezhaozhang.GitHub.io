---

title: "Encoding Rotated Images with Autoencoder"
date: 2023-09-15
typora-root-url: ./..
---



In this study, we explore the application of PyTorch-based autoencoders, featuring convolutional layers, to encode images from the Fashion-MNIST dataset. Our autoencoder effectively encodes and decodes original images. However, a notable challenge arises when we feed rotated images into the model. These rotated images are often decoded incorrectly and classified into different categories. We address this issue by training the model on a combination of original and randomly rotated images, enabling it to decode rotated input correctly.



### Data

We use a subset of the Fashion-MNIST dataset for our experiments, focusing on four out of the ten available classes. Below is the code snippet for loading the dataset and a sample of the images.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor,  RandomRotation

dataset_train = datasets.FashionMNIST(
    root='data',
    train=True,
    transform = ToTensor(),
    download=True)

dataset_test = datasets.FashionMNIST(
    root='data',
    train=False,
    transform = ToTensor(),
    download=True)
```

<figure>
  <center>
  <img src="/assets/images/autoencoder_image_random_sample_4classes.svg" width="400">
   </center>
  <center>
    <figcaption> Figure 1. Illustrative samples from the Fashion-MNIST dataset, showcasing four classes: Trouser, Dress, Sandal, and Bag. Three representative images are displayed for each class, with all images being grayscale and sized at 28x28 pixels.
    </figcaption>
  </center>
</figure>


### Autoencoder Model

The autoencoder comprises an encoder and decoder, both constructed using 2D convolutional layers. The encoded representation of an image consists of a pair of real numbers (x, y).

```python
class autoencoder(nn.Module):
    def __init__(self, input_channel):
        PAD = 1
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, 32, (3,3), stride=2, padding=PAD),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3), stride=2, padding=PAD),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3,3), stride=2, padding=PAD),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(2)
        )

        self.decoder = nn.Sequential(
            nn.LazyLinear(2048),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=(3,3), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32,  kernel_size=(3,3), stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channel,  kernel_size=(3,3), stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def decode(self, x):
        return self.decoder(x)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))
```

A summary of the model structure and parameters is provided in the code below.

```python
from torchsummary import summary
device = 'cuda' if torch.cuda.is_available() else 'cpu'
summary(autoencoder(1).to(device), (1,28, 28), device=device)
```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 14, 14]             320
              ReLU-2           [-1, 32, 14, 14]               0
            Conv2d-3             [-1, 64, 7, 7]          18,496
              ReLU-4             [-1, 64, 7, 7]               0
            Conv2d-5            [-1, 128, 4, 4]          73,856
              ReLU-6            [-1, 128, 4, 4]               0
           Flatten-7                 [-1, 2048]               0
            Linear-8                    [-1, 2]           4,098
            Linear-9                 [-1, 2048]           6,144
        Unflatten-10            [-1, 128, 4, 4]               0
  ConvTranspose2d-11             [-1, 64, 7, 7]          73,792
             ReLU-12             [-1, 64, 7, 7]               0
  ConvTranspose2d-13           [-1, 32, 14, 14]          18,464
             ReLU-14           [-1, 32, 14, 14]               0
  ConvTranspose2d-15            [-1, 1, 28, 28]             289
          Sigmoid-16            [-1, 1, 28, 28]               0
================================================================
Total params: 195,459
Trainable params: 195,459
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.38
Params size (MB): 0.75
Estimated Total Size (MB): 1.13
----------------------------------------------------------------
```



##### Training with Original Images

We initially train the model using only the original images. The training process involves minimizing the Mean Squared Error (MSE) loss between the input and the reconstructed output.

```python
model = autoencoder(1)
model.to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_epoches = 51
n_print = 10

loss_history = []
for epoch in range(n_epoches):
    loss_train = 0
    count_train = 0

    loss_test = 0
    count_test = 0

    # training
    model.train()
    for ds,_ in dl_train:
        x = ds.to(device)
        pred = model(x)
        loss = loss_fn(pred, x)
        loss_train += loss.item()*len(ds)
        count_train += len(ds)
        loss.backward()

        optimizer.step()

        optimizer.zero_grad()
    # testing
    model.eval()
    with torch.no_grad():
      for ds, _ in dl_test:
        x = ds.to(device)
        pred = model(x)
        loss_test += loss_fn(pred, x).item()*len(ds)
        count_test += len(ds)

    if epoch % n_print == 0:
        print(f"epoch={epoch}, train loss={loss_train/count_train}, test loss={loss_test/count_test}")
    loss_history.append([epoch, loss_train/count_train, loss_test/count_test, loss_train, count_train, loss_test,count_test])
```

Each image is encoded into a pair of real numbers denoted as $(x,y)$. Figure 2 illustrates the distribution of encoded images in the training dataset.

<figure>
  <center>
  <img src="/assets/images/autoencoder_embedding_classes.svg" width="800">
   </center>
  <center>
    <figcaption> Figure 2. Distribution of the encoded images in the training dataset.
    </figcaption>
  </center>
</figure>
Figure 3 displays sample images both before and after passing through the trained model. The autoencoder filters out specific image details during this process, preserving only the underlying general features.

<figure>
  <center>
  <img src="/assets/images/autoencoder_image-before-after-model.svg" width="800">
   </center>
  <center>
    <figcaption> Figure 3. Upper row: Images used as input to the model. Lower row: Images produced as output by the model.
    </figcaption>
  </center>
</figure>



##### Incorrect Decoding of Randomly Rotated Images

When we introduce randomly rotated images into the autoencoder, it often fails to decode them accurately. These rotated images are erroneously classified into different categories. Figure 4 presents examples of this phenomenon for each of the four classes. For instance, when rotated, a Trouser image may be decoded as a Bag, Dress, or Sandal. The same applies to Dress images being decoded as Bags. However, rotated Sandal and Bag images are decoded correctly, likely due to their distinct structural characteristics.

<figure>
  <center>
  <img src="/assets/images/autoencoder_decode_rotated_images_Trouser.svg" width="900">
    <img src="/assets/images/autoencoder_decode_rotated_images_Dress.svg" width="900">
    <img src="/assets/images/autoencoder_decode_rotated_images_Sandal.svg" width="900">
    <img src="/assets/images/autoencoder_decode_rotated_images_Bag.svg" width="900">
   </center>
  <center>
    <figcaption> Figure 4. The top row features randomly rotated images from each of the four classes, fed into the model. The corresponding decoded images are presented in the bottom row. Notably, the rotated images are incorrectly decoded in certain instances, leading to misclassification.
    </figcaption>
  </center>
</figure>




##### Training Model with Randomly Rotated Images

To address the issue of incorrect decoding, we train the model with randomly rotated images. For each image in the training dataset, five randomly rotated versions are added as inputs. The target image for both the original and rotated versions remains the same as the original image.

After this training, the autoencoder can accurately decode rotated images in the test dataset, as shown in Figure 5.

<figure>
  <center>
  <img src="/assets/images/autoencoder_model_trained_w_rotated_images_decode_rotated_images_Trouser.svg" width="900">
    <img src="/assets/images/autoencoder_model_trained_w_rotated_images_decode_rotated_images_Dress.svg" width="900">
    <img src="/assets/images/autoencoder_model_trained_w_rotated_images_decode_rotated_images_Sandal.svg" width="900">
    <img src="/assets/images/autoencoder_model_trained_w_rotated_images_decode_rotated_images_Bag.svg" width="900">
   </center>
  <center>
    <figcaption> Figure 5. In the top row, images from each of the four classes are randomly rotated at various angles and fed into the model trained with randomly rotated images. The resulting decoded images are displayed in the bottom row, showcasing that all rotated images are accurately decoded into their respective correct classes.
    </figcaption>
  </center>
</figure>



Figure 6 provides an analysis of the embeddings generated by the encoder. Each class is represented by one image and nine randomly rotated versions. It is evident that the classes are separated in the embedding space, and within each class, there is variability due to the different rotations.

<figure>
  <center>
  <img src="/assets/images/autoencoder_rotated_encoded_xyplot_annotated.svg" width="800">
   </center>
  <center>
    <figcaption> Figure 6. Image embeddings are visualized by ten random rotations of a single image representing each of the four classes.
    </figcaption>
  </center>
</figure>



### Summary

In conclusion, our autoencoder, featuring convolutional layers, successfully encodes images into a lower-dimensional representation. Initially, it struggles to decode randomly rotated images correctly. However, after training with these rotated images, the autoencoder can accurately decode. Each decoded image represents a distribution of random rotations, indicating the model's ability to generalize rotation invariance.



### References

Foster, D. (2023). *Generative Deep Learning* (2nd ed.). O'Reilly Media.

