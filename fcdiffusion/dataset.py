# # # # import json
# # # import cv2
# # # import numpy as np
# # # from torch.utils.data import Dataset


# # # class TrainDataset(Dataset):
# # #     def __init__(self):
# # #         self.data = []
# # #         with open('datasets/training_data.json', 'rt') as f:
# # #             for line in f:
# # #                 self.data.append(json.loads(line))

# # #     def __len__(self):
# # #         return len(self.data)

# # #     def __getitem__(self, idx):
# # #         item = self.data[idx]

# # #         img_path = item['img_path']
# # #         prompt = item['prompt']

# # #         img = cv2.imread(img_path)

# # #         # resize img to 512 x 512
# # #         img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

# # #         # Do not forget that OpenCV read images in BGR order.
# # #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # #         # Normalize images to [-1, 1].
# # #         img = (img.astype(np.float32) / 127.5) - 1.0

# # #         return dict(jpg=img, txt=prompt, path=img_path.split('6.5')[1])


# # # class TestDataset(Dataset):
# # #     def __init__(self, img_path: str, prompt: str, res_num: int):
# # #         self.data = []
# # #         self.img_path = img_path
# # #         self.prompt = prompt
# # #         self.res_num = res_num

# # #         for i in range(self.res_num):
# # #             self.data.append({'jpg': img_path,
# # #                               'txt': prompt})

# # #     def __len__(self):
# # #         return len(self.data)

# # #     def __getitem__(self, idx):
# # #         item = self.data[idx]

# # #         img_path = item['jpg']
# # #         prompt = item['txt']

# # #         img = cv2.imread(img_path)
# # #         img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
# # #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # #         img = (img.astype(np.float32) / 127.5) - 1.0

# # #         return dict(jpg=img, txt=prompt, path=img_path)

# # import json
# # import cv2
# # import numpy as np
# # from torch.utils.data import Dataset


# # # class TrainDataset(Dataset):
# # #     def __init__(self):
# # #         self.data = []
# # #         with open('datasets/training_data.json', 'rt') as f:
# # #             for line in f:
# # #                 self.data.append(json.loads(line))

# # #     def __len__(self):
# # #         return len(self.data)

# # #     def __getitem__(self, idx):
# # #         item = self.data[idx]

# # #         img_path = item['img_path']
# # #         prompt = item['prompt']

# # #         img = cv2.imread(img_path)
        
# # #         # resize img to 512 x 512
# # #         try:
# # #             img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

# # #             # Do not forget that OpenCV read images in BGR order.
# # #             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # #             # Normalize images to [-1, 1].
# # #             img = (img.astype(np.float32) / 127.5) - 1.0
# # #             return dict(jpg=img, txt=prompt, path=img_path.split('6.5')[1])
# # #         except Exception as e:
# # #             print(f"{img_path}:{e}")

# # class TrainDataset(Dataset):  
# #     def __init__(self):  
# #         self.data = []  
# #         try:  
# #             with open('/home/apulis-dev/userdata/FCDiffusion_code/datasets/training_data_new.json', 'rt') as f:  
# #                 for line in f:  
# #                     item = json.loads(line)  
# #                     if self._process_item(item):  
# #                         self.data.append(item)  
# #         except FileNotFoundError:  
# #             print("Training data file not found.")  
# #             raise  
  
# #     def _process_item(self, item):  
# #         img_path = item['img_path']  
# #         prompt = item['prompt']  
  
# #         try:  
# #             img = cv2.imread(img_path)  
# #             if img is None:  
# #                 print(f"Failed to load image: {img_path}")  
# #                 return False  
  
# #             # resize img to 512 x 512  
# #             img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)  
  
# #             # Convert BGR to RGB  
# #             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
  
# #             # Normalize images to [-1, 1].  
# #             img = (img.astype(np.float32) / 127.5) - 1.0  
  
# #             # Store the processed image and metadata in a dictionary  
# #             item['jpg'] = img  
# #             item['txt'] = prompt  
# #             item['path'] = img_path.split('images')[1]  
# #             return True  
# #         except Exception as e:  
# #             print(f"{img_path}: {e}")  
# #             return False  
  
# #     def __len__(self):  
# #         return len(self.data)  
  
# #     def __getitem__(self, idx):  
# #         return self.data[idx]

        


# # class TestDataset(Dataset):
# #     def __init__(self, img_path: str, prompt: str, res_num: int):
# #         self.data = []
# #         self.img_path = img_path
# #         self.prompt = prompt
# #         self.res_num = res_num

# #         for i in range(self.res_num):
# #             self.data.append({'jpg': img_path,
# #                               'txt': prompt})

# #     def __len__(self):
# #         return len(self.data)

# #     def __getitem__(self, idx):
# #         item = self.data[idx]

# #         img_path = item['jpg']
# #         prompt = item['txt']

# #         img = cv2.imread(img_path)
# #         img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
# #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #         img = (img.astype(np.float32) / 127.5) - 1.0

# #         return dict(jpg=img, txt=prompt, path=img_path)


# import json
# import cv2
# import numpy as np
# from torch.utils.data import Dataset
# from collections import OrderedDict


# class LimitedCache:
#     """
#     A fixed-size cache to store limited items, removing the oldest when the limit is exceeded.
#     """
#     def __init__(self, max_size=100):
#         self.cache = OrderedDict()
#         self.max_size = max_size

#     def get(self, key):
#         return self.cache.get(key, None)

#     def add(self, key, value):
#         if key in self.cache:
#             self.cache.move_to_end(key)
#         else:
#             self.cache[key] = value
#             if len(self.cache) > self.max_size:
#                 self.cache.popitem(last=False)


# class TrainDataset(Dataset):
#     """
#     Dataset for training. Images are loaded lazily with a limited-size cache to control memory usage.
#     """
#     def __init__(self, data_file: str, img_size: tuple = (512, 512), cache_size=100):
#         self.data = []
#         self.img_size = img_size
#         self.image_cache = LimitedCache(max_size=cache_size)

#         # Load metadata from the JSON file
#         try:
#             with open(data_file, 'rt') as f:
#                 for line in f:
#                     item = json.loads(line)
#                     if self._process_item(item):
#                         self.data.append(item)
#         except FileNotFoundError:
#             print("Training data file not found.")
#             raise

#     def _process_item(self, item):
#         """
#         Process metadata for an item. Validate and store image paths and prompts.
#         """
#         img_path = item['img_path']
#         prompt = item['prompt']

#         # Only store the necessary metadata
#         item['txt'] = prompt
#         item['path'] = img_path.split('6.5')[1]
#         item['img_path'] = img_path  # Store the full path for lazy loading
#         return True

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         """
#         Get a single item from the dataset. The image is loaded and cached on demand.
#         """
#         item = self.data[idx]
#         img_path = '../FCDiffusion_code-main/' + item['img_path']
#         prompt = item['txt']

#         img = self._load_image(img_path)

#         return dict(jpg=img, txt=prompt, path=item['path'])

#     def _load_image(self, img_path):
#         """
#         Load and preprocess an image, with caching to limit memory usage.
#         """
#         cached_img = self.image_cache.get(img_path)
#         if cached_img is not None:
#             return cached_img

#         img = cv2.imread(img_path)
#         if img is None:
#             raise FileNotFoundError(f"Image not found at {img_path}")

#         # Resize image to target size
#         img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)

#         # Convert BGR to RGB
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # Normalize image to range [-1, 1]
#         img = (img.astype(np.float32) / 127.5) - 1.0

#         # Add the image to the cache
#         self.image_cache.add(img_path, img)

#         return img


# class TestDataset(Dataset):
#     """
#     Dataset for testing. It uses the same image and prompt multiple times for testing purposes.
#     """
#     def __init__(self, img_path: str, prompt: str, res_num: int, img_size: tuple = (512, 512), cache_size=10):
#         self.data = []
#         self.img_path = img_path
#         self.prompt = prompt
#         self.res_num = res_num
#         self.img_size = img_size
#         self.image_cache = LimitedCache(max_size=cache_size)

#         # Create multiple entries of the same image and prompt
#         for i in range(self.res_num):
#             self.data.append({'jpg': img_path, 'txt': prompt})

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         img_path = item['jpg']
#         prompt = item['txt']

#         img = self._load_image(img_path)

#         return dict(jpg=img, txt=prompt, path=img_path)

#     def _load_image(self, img_path):
#         """
#         Load and preprocess an image, with caching to limit memory usage.
#         """
#         cached_img = self.image_cache.get(img_path)
#         if cached_img is not None:
#             return cached_img

#         img = cv2.imread(img_path)
#         if img is None:
#             raise FileNotFoundError(f"Image not found at {img_path}")

#         # Resize image to target size
#         img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)

#         # Convert BGR to RGB
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # Normalize image to range [-1, 1]
#         img = (img.astype(np.float32) / 127.5) - 1.0

#         # Add the image to the cache
#         self.image_cache.add(img_path, img)

#         return img




import json
import cv2
import numpy as np
from torch.utils.data import Dataset
from collections import OrderedDict
import torch


class LimitedCache:
    """
    A fixed-size cache to store limited items, removing the oldest when the limit is exceeded.
    """
    def __init__(self, max_size=100):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        return self.cache.get(key, None)

    def add(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)


class TrainDataset(Dataset):
    """
    Dataset for training. Images are loaded lazily with a limited-size cache to control memory usage.
    """
    def __init__(self, data_file: str, img_size: tuple = (512, 512), cache_size=100):
        self.data = []
        self.img_size = img_size
        self.image_cache = LimitedCache(max_size=cache_size)

        # Load metadata from the JSON file
        # data_file = data_file + '../DGM'
        try:
            with open(data_file, 'rt') as f:
                for line in f:
                    item = json.loads(line)
                    if self._process_item(item):
                        self.data.append(item)
        except FileNotFoundError:
            print("Training data file not found.")
            raise

    def _process_item(self, item):
        """
        Process metadata for an item. Validate and store image paths and prompts.
        """
        img_path = item['img_path']
        prompt = item['prompt']

        # Only store the necessary metadata
        item['txt'] = prompt
        # item['path'] = img_path.split('images')[1]
        # print(f"img_path:{img_path}")
        # item['path'] = img_path.split('6.5')[1]
        item['img_path'] = "/home/apulis-dev/userdata/DGM/" + img_path  # Store the full path for lazy loading
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset. The image is loaded and cached on demand.
        """
        item = self.data[idx]
        img_path = item['img_path']
        prompt = item['txt']

        img = self._load_image(img_path)
        return dict(jpg=img, txt=prompt, path=img_path)
        # return dict(jpg=img, txt=prompt, path=item['path'])
    # def __getitem__(self, idx):
    #     item = self.data[idx]
    #     img = self._load_image(item['img_path'])
    #     assert img.shape[-1] == 3, f"图像通道数错误，应为3，实际为{img.shape[-1]}"
    #     text = item['txt']
    #     return {
    #         'jpg': torch.from_numpy(img),  # Shape [H, W, C]
    #         'txt': text
    #     }

    def _load_image(self, img_path):
        """
        Load and preprocess an image, with caching to limit memory usage.
        """
        cached_img = self.image_cache.get(img_path)
        if cached_img is not None:
            return cached_img

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Resize image to target size
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize image to range [-1, 1]
        img = (img.astype(np.float32) / 127.5) - 1.0

        # Add the image to the cache
        self.image_cache.add(img_path, img)

        return img


class TestDataset(Dataset):
    """
    Dataset for testing. It uses the same image and prompt multiple times for testing purposes.
    """
    def __init__(self, img_path: str, prompt: str, res_num: int, img_size: tuple = (512, 512), cache_size=10):
        self.data = []
        self.img_path = img_path
        self.prompt = prompt
        self.res_num = res_num
        self.img_size = img_size
        self.image_cache = LimitedCache(max_size=cache_size)

        # Create multiple entries of the same image and prompt
        for i in range(self.res_num):
            self.data.append({'jpg': img_path, 'txt': prompt})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['jpg']
        prompt = item['txt']

        img = self._load_image(img_path)

        return dict(jpg=img, txt=prompt, path=img_path)

    def _load_image(self, img_path):
        """
        Load and preprocess an image, with caching to limit memory usage.
        """
        cached_img = self.image_cache.get(img_path)
        if cached_img is not None:
            return cached_img

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Resize image to target size
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize image to range [-1, 1]
        img = (img.astype(np.float32) / 127.5) - 1.0

        # Add the image to the cache
        self.image_cache.add(img_path, img)

        return img


