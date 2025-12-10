# Data Management in PyTorch

## Datasets

Often times the data that is provided to us may not be in a form that can be directly used in classes like DataLoader. For example the image names may be generic, and the labels might be in some other file (like a .mat file). In such cases. It is crucial that we learn how to define our own Datasets and Dataloaders.

## Data Access

### Defining our own Custom Datasets

```python
import os

class OxfordFlowersDataset(Dataset):

  def __init__(self, root_dir):
    self.root_dir = root_dir
    self.img_dir = os.path.join(root_dir, 'images')

    labels_matlab = scipy.io.loadmat(os.path.join(root_dir, 'imagelabels.mat'))

    self.labels = labels_matlab['labels'][0] - 1
```

Here the method adopted is called Lazy loading of Data, because if we initialize the class with data as it is, it uses up alot of RAM, which is unnecessary. Instead, we just mention where to find the data.

labels are adjusted by subtracting 1 because PyTorch expects that the class labels start from 0.

```python

def __len__(self):
  return len(self.labels)
```

Used for returning the total number of samples in the dataset.

```python
from PIL import Image

def __getitem__(self, idx):

  img_name = f'image_{idx+1:05d}.jpg'
  img_path = os.path.join(self.img_dir, img_name)

  image = Image.open(img_path)
  label = self.labels[idx]

  return image, label
```

This dunder function is used to return the image and its corresponding label for the index provided. img_name depends on the actual data in the directory. This only works for the image pattern in the actual dataset.

Also here idx is incremented by 1 because the dataset images start with image_00001. If 1 was not added, it would have taken image number 00000 (or image_00000) which does not exist.

So study the data, especially its metadata.

## Transform Pipelines (Quality)

### Learning why raw data won't work

Batching won't work because pytorch expects that the items in a batch are of same dimensions. Which is rarely the case for image data. Also, PyTorch expects tensors, not image data.

```python
transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
])
```

transforms.Resize(256) resizes the shorter edge to 256 whilst preserving the aspect ratio of the image. Hard resizing where we give both dimensions (256, 256) would distort the image.

Then transforms.CenterCrop(224) is used to obtain the middle portion (the 224x224 square image) of the image.

Now to convert the images into tensors:

```python
transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),        <------- Add this
  transforms.Normalize(mean= [...],
                        std= [...])
])
```

ToTensor() is called 'The tensor Bridge'. Before the bridge the data type is image. After the bridge, the data is tensor. So applying transforms that could only be applied to tensors to images would cause errors. So handle that properly.

Adding transforms to the OxfordFlowersDataset class:

```python
class OxfordFlowersDataset(Dataset):

  def __init__(self, root_dir, transform = None):
    # all other code
    self.transform = transform

  def __getitem__(self, idx):
    # all other code
    if self.transform:
      image = self.transform(image)
    return image, label
```

Now it could be batched:

```python
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```

For debugging, Take single datapoints and apply transforms individually.

## DataLoader

### Splitting the data

```python
train_dataset, val_dataset, test_dataset = random_split(
  dataset, [train_size, val_size, test_size]
)
```

This gives a good mix the entire data and distributes them according to the sizes mentioned.

### Batching using DataLoader

iterating through the dataloader object gives us batch-wise data.
For iterating through the first batch without starting a loop:

```python
images, label = next(iter(train_loader))
```

## Bug-proofing

### On-the-fly transformation of PyTorch

Random transforms are applied to the training dataset as it is loaded, without extra memory usages, so that the model see different versions of the same image each time.

```python
train_transform = transforms.Compose([
  #Random augmentation transforms
  transforms.RandomHorizontalFlip(p=0.5)
  transforms.RandomRotation(degrees=10),
  transforms.ColorJitter(brightness=0.2),

  #Other preprocessing steps
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean= [...],
                        std= [...])
])
```

### Corrupted files (Gracefully handling)

In **getitem** function include:

```python
image.verify()
image = Image.open(img_path)      <---- Reopen the image, because verify, closes the file.

if image.size[0] < 32 or image.size[1] < 32:
  raise ValueError(f"Image too small")

if image.mode != 'RGB':      <---- Converting to RGB
  image = image.convert('RGB')
```

In case of other Exceptions, take the next idx:

```python
next_idx = (idx + 1) % len(self)
return self.__getitem__(next_idx)
```

### Monitoring data

```python
def __getitem__(self, idx):
  import time
  start_time = time.time()

  self.access_counts[idx] = self.access_counts.get(idx, 0) + 1

  result = super().__getitem__(idx)

  load_time = time.time() - start_time
  self.load_times.append(load_time)

  if load_time > 1.0:
    print(f" Slow load for image index : {idx}"
          "Time taken: {load_time:.2f}s")
  return result
```
