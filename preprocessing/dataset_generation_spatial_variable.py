# This module contains the functions used for the dataset generation needed as input and target for the deep-learning model
# Compared to 'dataset_generation' module, this one loads datasets taking the best images for each low-flow season regardless of the month

import os
import torch 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

from osgeo import gdal
from torch.utils.data import TensorDataset 

from preprocessing.satellite_analysis_pre import count_pixels
from preprocessing.satellite_analysis_pre import load_avg

def load_image_array(path, scaled_classes=True):
    '''
    This function is used to load a single image using Gdal library. It also converts and returns it into a numpy array with dtype = np.float32.
    It is implemented and tested to work with JRC collection exported in grayscale (i.e., with pixel values between 0, 1 and 2).

    It also updates the pixels values by subtracting 1 from the whole image (element-wise operation) in order to later implement an algorithm that
    masks only non-water and water pixels (0 and 1 pixels, respectively) to train the model on these pixels only and neglect the no-data ones. 

    It can also scale the original pixel values by setting the new classes as follows:
            - no-data: -1
            - non-water: 0
            - water: 1

    Inputs: 
           path = str, contains full path of the image to be shown
           scaled_classes = bool, sets whether pixel classes are scaled to the range [-1, 1] or kept within the original one [0, 2]
                            default: True, pixel classes are scaled. 
    
    Output: 
           img_array = np.array, 2D array representing the loaded image
    '''
    img = gdal.Open(path)
    img_array = img.ReadAsArray().astype(np.float32)

    # scale the pixel value for each class with the updated classification
    if scaled_classes:

       img_array = img_array.astype(int)
       img_array[img_array==0] = -1
       img_array[img_array==1] = 0
       img_array[img_array==2] = 1
    
    return img_array

def create_dir_list(train_val_test, dir_folders=r'data\satellite\dataset', collection=r'JRC_GSW1_4_MonthlyHistory'):
    ''' 
    Get list of path of training, validation and testing datasets.

    Inputs:
           train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           dir_folders = str, directory where folders are stored
                         default: r'data\satellite\dataset'
           collection = str, specifies the satellite images collection.
                        default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
    
    Output:
           list_dir = list, contains paths of training, validation and testing dataset folders
    ''' 
    list_dir = []

    # get list of all folders
    for item in os.listdir(dir_folders):
        # get all existing directory
        if os.path.isdir(os.path.join(dir_folders, item)):
            # get only directories that match collection and usage 
            if (train_val_test in item) & (collection in item):
                list_dir.append(os.path.join(dir_folders, item))
    
    # sort by reach id
    list_dir.sort(key=lambda x: int(x.split(f'_{train_val_test}_r')[-1]))
    return list_dir

def create_list_images(train_val_test, reach, dir_folders=r'data\satellite\dataset', collection=r'JRC_GSW1_4_MonthlyHistory'):
    '''
    This function returns the paths of the satellite images present within a folder given use and reach. 
    It will be used later for loading and creating the dataset.

    Inputs:
           train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           reach = int, representing reach number. Number increases going upstream
                   For training, the available range is 1-28 (included)
           dir_folders = str, directory where folders are stored
                         default: r'data\satellite\dataset'
           collection = str, specifies the satellite images collection.
                        default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
    
    Outputs:
            list_dir_images = list, contains the path for each image of the dataset needed for loading it                     
    '''
    
    folder = os.path.join(str(dir_folders), collection + rf'_{train_val_test}_r{reach}')
    list_dir_images = []
    for image in os.listdir(folder):
        if image.endswith('.tif'):
            path_image = os.path.join(folder, image)
            list_dir_images.append(path_image)
    return list_dir_images

def create_datasets(train_val_test, reach, year_target=5, nodata_value=-1, dir_folders=r'data\satellite\dataset', 
                    collection=r'JRC_GSW1_4_MonthlyHistory', scaled_classes=True):
    '''
    This function creates the input and target dataset for each specific use and reach. It returns two lists of lists, with the input ones having n-elements 
    (with n depending on the year of prediction: if fifth year is predicted the list has four elements), while the target ones are one-element lists 
    (containing the year of prediction).

    It also reaplces `no-data` pixels with the season averages previously computed and stored in specific .csv files.
    
    Inputs: 
           train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           reach = int, representing reach number. Number increases going upstream.
                   For training, the available range is 1-28 (included)
           year_target = int, sets the year predicted after a sequence of input years.
                         default: 5, input dataset is made of 4 images and 5th year is the predicted one
           nodata_value = int, represents pixel value of no-data class.
                          default: -1, based on the updated pixel classes. 
                          If `scaled_classes` = False, this should be set to 0
           dir_folders = str, directory where folders are stored
                         default: r'data\satellite\dataset'
           collection = str, specifies the satellite images collection.
                        default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
           scaled_classes = bool, sets whether pixel classes are scaled to the range [-1, 1] or kept within the original one [0, 2]
                            default: True, pixel classes are scaled. 
    
    Outputs:
            input_dataset, target_dataset = lists of lists, contain the input and target images respectively  
    '''
        # list of images' paths
    list_dir_images = create_list_images(train_val_test, reach, dir_folders, collection)
    # list of images (arrays)
    images_array = [load_image_array(list_dir_images[i], scaled_classes=scaled_classes) for i in range(len(list_dir_images))]
    # load season averages
    avg_imgs = [load_avg(train_val_test, reach, 1988, dir_averages=r'data\satellite\averages')]
    for year in range(1988, 2021):
        avg_imgs.append(load_avg(train_val_test, reach, year, dir_averages=r'data\satellite\averages'))
    avg_imgs.append(load_avg(train_val_test, reach, 2021, dir_averages=r'data\satellite\averages'))
    # avg_imgs = [load_avg(train_val_test, reach, year, dir_averages=r'data\satellite\averages') for year in range(1988, 1988 + len(images_array)))]
    # print(len(images_array)), print(len(avg_imgs))
    # avg_imgs.append(load_avg(train_val_test, reach, 2021, dir_averages=r'data\satellite\averages'))
    # replace missing data
    good_images_array = [np.where(image==nodata_value, avg_imgs[i], image) for i, image in enumerate(images_array)]
    
    good_images_array = good_images_array[1:-1]

    input_dataset = []
    target_dataset = []
    
    # loop through images to append these in the originally empty lists
    for i in range(len(good_images_array)-year_target):
        input_dataset.append(good_images_array[i:i+year_target-1])
        target_dataset.append([good_images_array[i+year_target-1]])

    return input_dataset, target_dataset

def combine_datasets(train_val_test, reach, year_target=5, nonwater_threshold=480000, nodata_value=-1, nonwater_value=0,   
                     dir_folders=r'data\satellite\dataset', collection=r'JRC_GSW1_4_MonthlyHistory', scaled_classes=True):
    '''
    This function filters the images based on non-water amount of pixels threshold.  
    If the requirement is not met, the full combination inputs-target is discarded.
    It selects therefore the best images for training the model. After averaging, `no-data` pixels are replaced with the season and neighboours average.
    If The full image is composed of `no-data`, the resulting average wil be a fully `non-water` image. Hence the need to skip these images.

    Inputs:
           train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           reach = int, representing reach number. Number increases going upstream.
                   For training, the available range is 1-28 (included)
           year_target = int, sets the year predicted after a sequence of input years.
                         default: 5, input dataset is made of 4 images and 5th year is the predicted one
           nonwater_threshold = int, min amount of non water pixels needed in the images - needed for input dataset only
                                default: 480000 
           nodata_value = int, represents pixel value of no-data class.
                          default: -1, based on the updated pixel classes. 
                          If `scaled_classes` = False, this should be set to 0
           nonwater_value = int, represents pixel value of water class.
                         default: 0, based on the updated pixel classes. 
                          If `scaled_classes` = False, this should be set to 1
           dir_folders = str, directory where folders are stored
                         default: r'data\satellite\dataset'
           collection = str, specifies the satellite images collection.
                        default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
           scaled_classes = bool, sets whether pixel classes are scaled to the range [-1, 1] or kept within the original one [0, 2]
                            default: True, pixel classes are scaled.
                        
    Output:
           filtered_input_dataset, filtered_target_dataset = lists, contain adequate images combinations for input and atrget dataset, respectively,
                                                             based on no-data and water thresholds
    '''
    input_dataset, target_dataset = create_datasets(train_val_test, reach, year_target, nodata_value, dir_folders, collection, scaled_classes)

    filtered_input_dataset, filtered_target_dataset = [], []
    # filter pairs based on the specified threshold
    for input_images, target_image in zip(input_dataset, target_dataset):
        input_combs = []
        # check input images
        for img in input_images:
            nonwater_count = count_pixels(img, nonwater_value) < nonwater_threshold 
            input_combs.append(nonwater_count)
            # check if input images are all suitable
            
        if all(input_combs):
            # check target images
            target_nonwater_thr = count_pixels(target_image[0], nonwater_value) < nonwater_threshold
            if target_nonwater_thr:
                # convert input images to tensor
                input_tensor = [img for img in input_images]
                # convert target image to tensor
                target_tensor = target_image[0]
                
                filtered_input_dataset.append(input_tensor)
                filtered_target_dataset.append(target_tensor)
    return filtered_input_dataset, filtered_target_dataset

# -------------------------- #
# old implementation #
# -------------------------- #

# def combine_datasets(train_val_test, reach, year_target=5, nodata_threshold=400000, nonwater_threshold=450000,
#                      water_threshold=30000, nodata_value = -1, nonwater_value=0, water_value = 1,   
#                      dir_folders=r'data\satellite\dataset', collection=r'JRC_GSW1_4_MonthlyHistory',
#                      scaled_classes=True):
#     '''
#     UPDATE DOCUMENTATION
#     This function filters the images based on no-data and water amount of pixels thresholds. If one of these requirements are not met,
#     either for any input image or for the target image, the full combination inputs-target is discarded.
#     It selects therefore the best images for training the model.

#     Inputs:
#            train_val_test = str, specifies for what the images are used for.
#                             available options: 'training', 'validation' and 'testing'
#            reach = int, representing reach number. Number increases going upstream.
#                    For training, the available range is 1-28 (included)
#            year_target = int, sets the year predicted after a sequence of input years.
#                          default: 5, input dataset is made of 4 images and 5th year is the predicted one
#            nodata_threshold = int, max amount of no-data pixels allowed in the images - needed for target dataset only
#                               default: 400000
#            nonwater_threshold = int, min amount of non water pixels needed in the images - needed for input dataset only
#                                 default: 450000 
#            water_threshold = int, min amount of water pixels needed in the images
#                               default: 30000
#            nodata_value = int, represents pixel value of no-data class.
#                           default: -1, based on the updated pixel classes. 
#                           If `scaled_classes` = False, this should be set to 0
#            nonwater_value = int, represents pixel value of water class.
#                          default: 0, based on the updated pixel classes. 
#                           If `scaled_classes` = False, this should be set to 1
#            water_value = int, represents pixel value of water class.
#                          default: 1, based on the updated pixel classes. 
#                           If `scaled_classes` = False, this should be set to 2
#            dir_folders = str, directory where folders are stored
#                          default: r'data\satellite\dataset'
#            collection = str, specifies the satellite images collection.
#                         default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
#            scaled_classes = bool, sets whether pixel classes are scaled to the range [-1, 1] or kept within the original one [0, 2]
#                             default: True, pixel classes are scaled.
                        
#     Output:
#            filtered_input_dataset, filtered_target_dataset = lists, contain adequate images combinations for input and atrget dataset, respectively,
#                                                              based on no-data and water thresholds
#     '''
#     input_dataset, target_dataset = create_datasets(train_val_test, reach, year_target, dir_folders, collection, scaled_classes)

#     filtered_input_dataset = []
#     filtered_target_dataset = []

#         # Filtering pairs based on the specified threshold
#     for input_images, target_image in zip(input_dataset, target_dataset):
#         input_combs = []
#         # check input images
#         for img in input_images:
#             nonwater_count = count_pixels(img, nonwater_value) < nonwater_threshold
#             water_count = count_pixels(img, water_value) > water_threshold
#             input_combination = True if (nonwater_count or water_count) else False  
#             input_combs.append(input_combination)
#             # print(f'input combs{input_combs}', nonwater_count, water_count)
#         # input_nonwater_thr = all(count_pixels(img, nonwater_value) < nonwater_threshold for img in input_images)
#         # input_water_thr = all(count_pixels(img, water_value) > water_threshold for img in input_images)
#         # input_combination = input_nonwater_thr or input_water_thr
        
#         # check target images
#         target_nodata_thr = count_pixels(target_image[0], nodata_value) < nodata_threshold
#         target_water_thr = count_pixels(target_image[0], water_value) > water_threshold
#         target_combination = target_nodata_thr or target_water_thr
#         # print('target ', target_nodata_thr, target_water_thr, target_combination)
#         # print(input_nonwater_thr, input_water_thr, target_nodata_thr, target_water_thr)
#         # apply both conditions
#         # i = 0
#         if all(input_combs) and target_combination:
#             # Convert input images to tensor
#             input_tensor = [img for img in input_images]
#             # Convert target image to tensor
#             target_tensor = target_image[0]
#             # Append as tuple
#             filtered_input_dataset.append(input_tensor)
#             filtered_target_dataset.append(target_tensor)
#         #     i =+ 1
#         #     print(i, ' sample')
#         # print('\n')
#     # print(f'Input {len(input_tensor[0])}')
#     # print(f'Target {len(target_tensor)}\n')
#     # pairs = [(input_seq, target) for input_seq, target in zip(filtered_input_dataset, filtered_target_dataset)]
#     return filtered_input_dataset, filtered_target_dataset

# -------------------------- #
# even older implementation #
# -------------------------- #

# def combine_datasets(train_val_test, reach, year_target=5, nodata_threshold=400000, nonwater_threshold=450000,
#                      water_threshold=30000, nodata_value = -1, nonwater_value=0, water_value = 1,   
#                      dir_folders=r'data\satellite\dataset', collection=r'JRC_GSW1_4_MonthlyHistory',
#                      scaled_classes=True):
#     '''
    # UPDATE DOCUMENTATION
#     This function filters the images based on no-data and water amount of pixels thresholds. If one of these requirements are not met,
#     either for any input image or for the target image, the full combination inputs-target is discarded.
#     It selects therefore the best images for training the model.

#     Inputs:
#            train_val_test = str, specifies for what the images are used for.
#                             available options: 'training', 'validation' and 'testing'
#            reach = int, representing reach number. Number increases going upstream.
#                    For training, the available range is 1-28 (included)
#            year_target = int, sets the year predicted after a sequence of input years.
#                          default: 5, input dataset is made of 4 images and 5th year is the predicted one
#            nodata_threshold = int, max amount of no-data pixels allowed in the images - needed for target dataset only
#                               default: 400000
#            nonwater_threshold = int, min amount of non water pixels needed in the images - needed for input dataset only
#                                 default: 450000 
#            water_threshold = int, min amount of water pixels needed in the images
#                               default: 30000
#            nodata_value = int, represents pixel value of no-data class.
#                           default: -1, based on the updated pixel classes. 
#                           If `scaled_classes` = False, this should be set to 0
#            nonwater_value = int, represents pixel value of water class.
#                          default: 0, based on the updated pixel classes. 
#                           If `scaled_classes` = False, this should be set to 1
#            water_value = int, represents pixel value of water class.
#                          default: 1, based on the updated pixel classes. 
#                           If `scaled_classes` = False, this should be set to 2
#            dir_folders = str, directory where folders are stored
#                          default: r'data\satellite\dataset'
#            collection = str, specifies the satellite images collection.
#                         default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
#            scaled_classes = bool, sets whether pixel classes are scaled to the range [-1, 1] or kept within the original one [0, 2]
#                             default: True, pixel classes are scaled.
                        
#     Output:
#            filtered_input_dataset, filtered_target_dataset = lists, contain adequate images combinations for input and atrget dataset, respectively,
#                                                              based on no-data and water thresholds
#     '''
#     input_dataset, target_dataset = create_datasets(train_val_test, reach, year_target, dir_folders, collection, scaled_classes)

#     filtered_input_dataset = []
#     filtered_target_dataset = []

#     # Filtering pairs based on the specified threshold
#     for input_images, target_image in zip(input_dataset, target_dataset):
#         # check input images
#         input_nonwater_thr = all(count_pixels(img, nonwater_value) < nonwater_threshold for img in input_images)
#         input_water_thr = all(count_pixels(img, water_value) > water_threshold for img in input_images)
#         input_combination = input_nonwater_thr or input_water_thr
#         # check target images
#         target_nodata_thr = count_pixels(target_image[0], nodata_value) < nodata_threshold
#         target_water_thr = count_pixels(target_image[0], water_value) > water_threshold
#         target_combination = target_nodata_thr or target_water_thr
#         # print(input_nonwater_thr, input_water_thr, target_nodata_thr, target_water_thr)
#         # apply both conditions
#         if input_combination and target_combination:
#             # Convert input images to tensor
#             input_tensor = [img for img in input_images]
#             # Convert target image to tensor
#             target_tensor = target_image[0]
#             # Append as tuple
#             filtered_input_dataset.append(input_tensor)
#             filtered_target_dataset.append(target_tensor)    
    
#     # print(f'Input {len(input_tensor[0])}')
#     # print(f'Target {len(target_tensor)}\n')
#     # pairs = [(input_seq, target) for input_seq, target in zip(filtered_input_dataset, filtered_target_dataset)]
#     return filtered_input_dataset, filtered_target_dataset

def create_full_dataset(train_val_test, year_target=5, nonwater_threshold=480000, nodata_value=-1, nonwater_value = 0, dir_folders=r'data\satellite\dataset', 
                        collection=r'JRC_GSW1_4_MonthlyHistory', scaled_classes=True, device='cuda:0', dtype=torch.int64):
    '''
    This function generates the full dataset for the given use, combining all reaches. It is built upon the function `combine_datasets`, 
    which creates the paired couples input-target for a specific reach and given a specified value of non-water pixels.

    It stacks all different pairs within one use in order to have the dataset ready for the training, validation and testing of the model.

    Inputs:
           train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           year_target = int, sets the year predicted after a sequence of input years.
                         default: 5, input dataset is made of 4 images and 5th year is the predicted one
           nonwater_threshold = int, minimum amount of water pixels accepted for the training, validation and testing
                                default: 480000, based on pixels distribution 
           nodata_value = int, represents pixel value of water class.
                          default: -1, based on the updated pixel classes. 
                          If `scaled_classes` = False, this should be set to 0
           nonwater_value = int, represents pixel value of water class.
                            default: 0, based on the updated pixel classes. 
                            If `scaled_classes` = False, this should be set to 1
           dir_folders = str, directory where folders are stored
                         default: r'data\satellite\dataset'
           collection = str, specifies the satellite images collection.
                        default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
           scaled_classes = bool, sets whether pixel classes are scaled to the range [-1, 1] or kept within the original one [0, 2]
                            default: True, pixel classes are scaled.
           device = str, , specifies device where memory is allocated for performing the computations
                    default: 'cuda' (GPU), other availble option: 'cpu'
           dtype = class, specifies the data type for torch.tensor method.
                   default: torch.int64, it also accepts `torch.float32` to allow gradient computation and backpropagation
    
    Output:
           dataset = TensorDataset, contains all coupled input-target samples for each reach and use
    '''
    # initialize stacked dictionaries
    stacked_dict = {'input': [], 'target': []}
    for folder in os.listdir(dir_folders):
        if train_val_test in folder:
            reach_id = folder.split('_r',1)[1]
            inputs, target = combine_datasets(train_val_test, int(reach_id), year_target, nonwater_threshold, 
                                              nodata_value, nonwater_value, dir_folders, collection, scaled_classes)
            stacked_dict['input'].extend(inputs)
            stacked_dict['target'].extend(target)
    
    # unsqueeze is done in order to have shape [n_samples, 4, 1, 1000, 500] for inputs
    # and [n_samples, 1, 1000, 500] for targets 
    
    # create tensors
    if dtype == None:
        input_tensor = torch.tensor(stacked_dict['input'], device=device)
        target_tensor = torch.tensor(stacked_dict['target'], device=device)
    else:
        input_tensor = torch.tensor(stacked_dict['input'], dtype=dtype, device=device)
        target_tensor = torch.tensor(stacked_dict['target'], dtype=dtype, device=device)
    
    dataset = TensorDataset(input_tensor, target_tensor) #.unsqueeze(2), .unsqueeze(1)

    return dataset

# ----------------------------------------- # 
# TEMPORAL SPLIT #
# ----------------------------------------- # 

def split_list(train_val_test, reach, month, year_end_train=2009, year_end_val=2015, dir_folders=r'data\satellite', collection=r'JRC_GSW1_4_MonthlyHistory'):
    '''
    UPDATE DOCUMENTATION
    This function is used to split the list created for the dataset generation into three sub-lists, training, validation and testing, respectively.
    It splits the lists by years, given the `year_end_train` and `year_end_val` values.
    
    This split is used for generating a temporal and no longer spatial dataset.
    
    Inputs:
           train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           reach = int, representing reach number. Number increases going upstream
                   For training, the available range is 1-28 (included)
           month = int, represents month of the year from which images are taken. 
                   Available options: 1, 2, 3 or 4 (non-monsoon season months with low flow conditions). 
                   If another value is given an Exception is raised.
           year_end_train = int, sets last images year used for the training
                            default: 2011, recommended to have 23 years of training data
           year_end_val = int, sets last images year used for the validation
                          default: 2016, recommended to have 5 years of both validation and testing data
           dir_folders = str, directory where datasets are stored
                         default: r'data\satellite'
           collection = str, specifies the satellite images collection.
                        default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
    
    Output:
           train_list, val_list, test_list = lists, contain the path for each training, validation and testing image of the dataset needed for loading it  
    '''
    if month not in {1, 2, 3, 4}:
        raise Exception(f'The specified month is {month}, which is not allowed. It can be either 1, 2, 3 or 4.')
    
    # get monhtly dataset directory
    dir_dataset = os.path.join(dir_folders, fr'dataset_month{month}')    
    list = create_list_images(train_val_test, reach, dir_folders=dir_dataset, collection=collection)
    
    # initialize lists
    train_list, val_list, test_list = [], [], [] 
    for path in list:
        # get image year
        year = int((path.split('\\')[-1]).split('_')[0])
        # training list
        if year <= year_end_train:
            train_list.append(path)
        # validation list
        elif year_end_train < year <= year_end_val:
            val_list.append(path)
        # testing list
        elif year > year_end_val:
            test_list.append(path)
    return train_list, val_list, test_list

def create_split_datasets(train_val_test, reach, month, use_dataset, year_end_train=2009, year_end_val=2015, year_target=5, nodata_value=-1,
                          dir_folders=r'data\satellite', collection=r'JRC_GSW1_4_MonthlyHistory', scaled_classes=True):
    '''
    This function creates the input and target dataset for each specific use and reach. It returns two lists of lists, with the input ones having n-elements 
    (with n depending on the year of prediction: if fifth year is predicted the list has four elements), while the target ones are one-element lists 
    (containing the year of prediction).

    It also replaces missing data with season and neighbours average.

    ATTENTION: this function can load images by using either `show_image_array` or `load_image`. Currently the former is used.
    The main difference in these functions are the pixel classes:
            - `show_image_array` can return both the original classification [0, 1, 2] and the scaled one [-1, 0, 1]
               for no-data, non-water and water - images are loaded with gdal library (faster) with uint8 datatype, 
               but if classification is scaled the dtype is set to numpy.int
            - `load_image` returns the scaled classification [-1, 0, 1] for no-data, non-water and water and images can be either
               numpy arrays or tensors. It uses the PIL library (slower) and returns numpy.int dtype arrays.
    
    Inputs: 
           train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           reach = int, representing reach number. Number increases going upstream.
                   For training, the available range is 1-28 (included)
           month = int, represents month of the year from which images are taken. 
                   Available options: 1, 2, 3 or 4 (non-monsoon season months with low flow conditions). 
           use_dataset = str, train_val_test 
           year_end_train = int, sets last images year used for the training
                            default: 2011, recommended to have 23 years of training data
           year_end_val = int, sets last images year used for the validation
                          default: 2016, recommended to have 5 years of both validation and testing data
           year_target = int, sets the year predicted after a sequence of input years.
                         default: 5, input dataset is made of 4 images and 5th year is the predicted one
           nodata_value = int, represents pixel value of water class.
                          default: -1, based on the updated pixel classes. 
                          If `scaled_classes` = False, this should be set to 0
           dir_folders = str, directory where folders are stored
                         default: r'data\satellite\dataset'
           collection = str, specifies the satellite images collection.
                        default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
           scaled_classes = bool, sets whether pixel classes are scaled to the range [-1, 1] or kept within the original one [0, 2]
                            default: True, pixel classes are scaled. 
    
    Outputs:
            input_dataset, target_dataset = lists of lists, contain the input and target images respectively  
    '''
    # list of images' paths
    train_list, val_list, test_list = split_list(train_val_test, reach, month, year_end_train, year_end_val, dir_folders, collection)
    
    if use_dataset == 'training':
        # list of images (arrays)
        images_train = [load_image_array(train_list[i], scaled_classes=scaled_classes) for i in range(len(train_list))]
        avg_train = [load_avg(train_val_test, reach, year, dir_averages=r'data\satellite\averages') for year in range(1988, 1988 + len(images_train))]
        # replace missing data
        good_images_train = [np.where(image==nodata_value, avg_train[i], image) for i, image in enumerate(images_train)]
        input_train, target_train = [], []
        for i in range(len(good_images_train)-year_target):
            input_train.append(good_images_train[i:i+year_target-1])
            target_train.append([good_images_train[i+year_target-1]])
        train = [input_train, target_train] 
        return train
    
    elif use_dataset == 'validation':
        images_val = [load_image_array(val_list[i], scaled_classes=scaled_classes) for i in range(len(val_list))]
        avg_val = [load_avg(train_val_test, reach, year, dir_averages=r'data\satellite\averages') for year in range(year_end_train, year_end_train + len(images_val))]
        good_images_val = [np.where(image==nodata_value, avg_val[i], image) for i, image in enumerate(images_val)]
        input_val, target_val = [], [] 
        for i in range(len(good_images_val)-year_target):
            input_val.append(good_images_val[i:i+year_target-1])
            target_val.append([good_images_val[i+year_target-1]])
        val = [input_val, target_val] 
        return val

    elif use_dataset == 'testing':
        images_test = [load_image_array(test_list[i], scaled_classes=scaled_classes) for i in range(len(test_list))]
        avg_test = [load_avg(train_val_test, reach, year, dir_averages=r'data\satellite\averages') for year in range(year_end_val, year_end_val + len(images_test))]
        good_images_test = [np.where(image==nodata_value, avg_test[i], image) for i, image in enumerate(images_test)]
        input_test, target_test = [], []
        for i in range(len(good_images_test)-year_target):
            input_test.append(good_images_test[i:i+year_target-1])
            target_test.append([good_images_test[i+year_target-1]])
        test = [input_test, target_test]
        return test
    
    else:
        raise Exception(f'The given use_dataset is {use_dataset} but is wrong.\n\
The possible choices are "training", "validation", "testing".')

def combine_split_datasets(train_val_test, reach, month, use_dataset, year_end_train=2009, year_end_val=2015, year_target=5, nonwater_threshold=480000, 
                           nodata_value = -1, nonwater_value=0, dir_folders=r'data\satellite', collection=r'JRC_GSW1_4_MonthlyHistory', scaled_classes=True):
    '''
    This function filters the images based on non-water amount of pixels threshold. 
    If the requirement is not met, the full combination inputs-target is discarded.
    It selects therefore the best images for training the model.

    Inputs:
           train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           reach = int, representing reach number. Number increases going upstream.
                   For training, the available range is 1-28 (included)
           year_target = int, sets the year predicted after a sequence of input years.
                         default: 5, input dataset is made of 4 images and 5th year is the predicted one
           nonwater_threshold = int, max amount of no-data pixels allowed in the images
                                default: 480000
           nodata_value = int, represents pixel value of no-data class.
                          default: -1, based on the updated pixel classes. 
                          If `scaled_classes` = False, this should be set to 0
           nonwater_value = int, represents pixel value of water class.
                            default: 0, based on the updated pixel classes. 
                            If `scaled_classes` = False, this should be set to 1
           dir_folders = str, directory where folders are stored
                         default: r'data\satellite\dataset'
           collection = str, specifies the satellite images collection.
                        default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
           scaled_classes = bool, sets whether pixel classes are scaled to the range [-1, 1] or kept within the original one [0, 2]
                            default: True, pixel classes are scaled.
                        
    Output:
           filtered_input_dataset, filtered_target_dataset = lists, contain adequate images combinations for input and atrget dataset, respectively,
                                                             based on no-data and water thresholds
    '''
    dataset = create_split_datasets(train_val_test, reach, month, use_dataset, year_end_train, year_end_val, 
                                             year_target, nodata_value, dir_folders, collection, scaled_classes)
    
    # get inputs and targets
    input_dataset, target_dataset = dataset[0], dataset[1]

    filtered_inputs, filtered_targets = [], []

    # filtering pairs based on the specified threshold
    for input_images, target_image in zip(input_dataset, target_dataset):
        input_combs = []
        # check input images
        for img in input_images:
            nonwater_count = count_pixels(img, nonwater_value) < nonwater_threshold 
            input_combs.append(nonwater_count)
            # check if input images are all suitable
        if all(input_combs):
            # check target images
            target_nonwater_thr = count_pixels(target_image[0], nonwater_value) < nonwater_threshold
            if target_nonwater_thr:
                # convert input images to tensor
                input_tensor = [img for img in input_images]
                # convert target image to tensor
                target_tensor = target_image[0]
                
                filtered_inputs.append(input_tensor)
                filtered_targets.append(target_tensor)
      
    filtered_dataset = [filtered_inputs, filtered_targets]
    return filtered_dataset

# -------------------------- #
# old implementation #
# -------------------------- #

# def combine_split_datasets(train_val_test, reach, month, year_end_train=2009, year_end_val=2015, year_target=5, nodata_threshold=400000, 
#                            nonwater_threshold=450000, water_threshold=30000, nodata_value = -1, nonwater_value=0, water_value = 1, 
#                            dir_folders=r'data\satellite', collection=r'JRC_GSW1_4_MonthlyHistory', scaled_classes=True):
#     '''
#     This function filters the images based on no-data and water amount of pixels thresholds. If one of these requirements are not met,
#     either for any input image or for the target image, the full combination inputs-target is discarded.
#     It selects therefore the best images for training the model.

#     Inputs:
#            train_val_test = str, specifies for what the images are used for.
#                             available options: 'training', 'validation' and 'testing'
#            reach = int, representing reach number. Number increases going upstream.
#                    For training, the available range is 1-28 (included)
#            year_target = int, sets the year predicted after a sequence of input years.
#                          default: 5, input dataset is made of 4 images and 5th year is the predicted one
#            nodata_threshold = int, max amount of no-data pixels allowed in the images
#                               default: 400000
#            water_threshold = int, min amount of water pixels needed in the images
#                               default: 12000
#            nodata_value = int, represents pixel value of no-data class.
#                           default: -1, based on the updated pixel classes. 
#                           If `scaled_classes` = False, this should be set to 0
#            water_value = int, represents pixel value of water class.
#                          default: 1, based on the updated pixel classes. 
#                           If `scaled_classes` = False, this should be set to 2
#            dir_folders = str, directory where folders are stored
#                          default: r'data\satellite\dataset'
#            collection = str, specifies the satellite images collection.
#                         default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
#            scaled_classes = bool, sets whether pixel classes are scaled to the range [-1, 1] or kept within the original one [0, 2]
#                             default: True, pixel classes are scaled.
                        
#     Output:
#            filtered_input_dataset, filtered_target_dataset = lists, contain adequate images combinations for input and atrget dataset, respectively,
#                                                              based on no-data and water thresholds
#     '''
#     train, val, test = create_split_datasets(train_val_test, reach, month, year_end_train, year_end_val, year_target, dir_folders, collection, scaled_classes)
        
#     # get inputs and targets for all uses
#     input_train, target_train = train[0], train[1]
#     input_val, target_val = val[0], val[1]
#     input_test, target_test = test[0], test[1]

#     filtered_input_train, filtered_target_train = [], []
#     filtered_input_val, filtered_target_val = [], []
#     filtered_input_test, filtered_target_test = [], []

#     # training - filtering pairs based on the specified threshold
#     for input_images, target_image in zip(input_train, target_train):
#         # check input images
#         input_nonwater_thr = all(count_pixels(img, nonwater_value) < nonwater_threshold for img in input_images)
#         input_water_thr = all(count_pixels(img, water_value) > water_threshold for img in input_images)
#         input_combination = input_nonwater_thr or input_water_thr
#         # check target images
#         target_nodata_thr = count_pixels(target_image[0], nodata_value) < nodata_threshold
#         target_water_thr = count_pixels(target_image[0], water_value) > water_threshold
#         target_combination = target_nodata_thr or target_water_thr
#         # print(input_nonwater_thr, input_water_thr, target_nodata_thr, target_water_thr)
#         # apply both conditions
#         if input_combination and target_combination:
#             # Convert input images to tensor
#             input_tensor = [img for img in input_images]
#             # Convert target image to tensor
#             target_tensor = target_image[0]
#             # Append as tuple
#             filtered_input_train.append(input_tensor)
#             filtered_target_train.append(target_tensor)    
    
#     # validation - filtering pairs based on the specified threshold
#     for input_images, target_image in zip(input_val, target_val):
#         # check input images
#         input_nonwater_thr = all(count_pixels(img, nonwater_value) < nonwater_threshold for img in input_images)
#         input_water_thr = all(count_pixels(img, water_value) > water_threshold for img in input_images)
#         input_combination = input_nonwater_thr or input_water_thr
#         # check target images
#         target_nodata_thr = count_pixels(target_image[0], nodata_value) < nodata_threshold
#         target_water_thr = count_pixels(target_image[0], water_value) > water_threshold
#         target_combination = target_nodata_thr or target_water_thr
#         # print(input_nonwater_thr, input_water_thr, target_nodata_thr, target_water_thr)
#         # apply both conditions
#         if input_combination and target_combination:
#             # Convert input images to tensor
#             input_tensor = [img for img in input_images]
#             # Convert target image to tensor
#             target_tensor = target_image[0]
#             # Append as tuple
#             filtered_input_val.append(input_tensor)
#             filtered_target_val.append(target_tensor)     

#     # testing - filtering pairs based on the specified threshold
#     for input_images, target_image in zip(input_test, target_test):
#         # check input images
#         input_nonwater_thr = all(count_pixels(img, nonwater_value) < nonwater_threshold for img in input_images)
#         input_water_thr = all(count_pixels(img, water_value) > water_threshold for img in input_images)
#         input_combination = input_nonwater_thr or input_water_thr
#         # check target images
#         target_nodata_thr = count_pixels(target_image[0], nodata_value) < nodata_threshold
#         target_water_thr = count_pixels(target_image[0], water_value) > water_threshold
#         target_combination = target_nodata_thr or target_water_thr
#         # print(input_nonwater_thr, input_water_thr, target_nodata_thr, target_water_thr)
#         # apply both conditions
#         if input_combination and target_combination:
#             # Convert input images to tensor
#             input_tensor = [img for img in input_images]
#             # Convert target image to tensor
#             target_tensor = target_image[0]
#             # Append as tuple
#             filtered_input_test.append(input_tensor)
#             filtered_target_test.append(target_tensor)    
        
#     filtered_train = [filtered_input_train, filtered_target_train]
#     filtered_val = [filtered_input_val, filtered_target_val]
#     filtered_test = [filtered_input_test, filtered_target_test]
#     return filtered_train, filtered_val, filtered_test

def create_split_dataset(month, use_dataset, year_target=5, year_end_train=2009, year_end_val=2015, nonwater_threshold=480000, nodata_value = -1, nonwater_value = 0, 
                         dir_folders=r'data\satellite', collection=r'JRC_GSW1_4_MonthlyHistory', scaled_classes=True, device='cuda:0', dtype=torch.int64):
    '''
    This function generates the full dataset for the given use, combining all reaches. It is built upon the function `combine_datasets`, 
    which creates the paired couples input-target for a specific reach and given a specified value of no-data and water pixels thresholds.

    It stacks all different pairs within one use in order to have the dataset ready for the training, validation and testing of the model.

    Inputs:
           train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           year_target = int, sets the year predicted after a sequence of input years.
                         default: 5, input dataset is made of 4 images and 5th year is the predicted one
           nodata_threshold = int, maximum amount of no-data pixels accepted for the training, validation and testing
                              default: 400000, quite high compared to total number of pixels (500000) but best images were already filtered out 
           water_threshold = int, minimum amount of water pixels accepted for the training, validation and testing
                             default: 12000, based on pixels distribution 
           nodata_value = int, represents pixel value of no-data class.
                          default: -1, based on the updated pixel classes. 
                          If `scaled_classes` = False, this should be set to 0
           water_value = int, represents pixel value of water class.
                         default: 1, based on the updated pixel classes. 
                          If `scaled_classes` = False, this should be set to 2
           dir_folders = str, directory where folders are stored
                         default: r'data\satellite\dataset'
           collection = str, specifies the satellite images collection.
                        default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
           scaled_classes = bool, sets whether pixel classes are scaled to the range [-1, 1] or kept within the original one [0, 2]
                            default: True, pixel classes are scaled.
           device = str, , specifies device where memory is allocated for performing the computations
                    default: 'cuda' (GPU), other availble option: 'cpu'
           dtype = class, specifies the data type for torch.tensor method.
                   default: torch.int64, it also accepts `torch.float32` to allow gradient computation and backpropagation
                   Other available options: none at the moment (03/05/2024), in case an update of this documentation and function is needed
    
    Output:
           dataset = TensorDataset, contains all coupled input-target samples for each reach and use
    '''
    train_val_test = ['training', 'validation', 'testing']
    dir_dataset = os.path.join(dir_folders, rf'dataset_month{month}')
    stacked_dataset= {'input': [], 'target': []}
    
    # loop over folder and use 
    for folder in os.listdir(dir_dataset):
        for use in train_val_test:
            if use in folder:
                reach_id = folder.split('_r',1)[1]
                filtered_dataset = combine_split_datasets(
                    use, reach_id, month, use_dataset, year_end_train, year_end_val, 
                    year_target, nonwater_threshold, nodata_value, nonwater_value,  
                    dir_folders, collection, scaled_classes
                    )
    
                stacked_dataset['input'].extend(filtered_dataset[0])
                stacked_dataset['target'].extend(filtered_dataset[1])
       
    inputs, targets = torch.tensor(stacked_dataset['input'], dtype=dtype, device=device), torch.tensor(stacked_dataset['target'], dtype=dtype, device=device)
    dataset = TensorDataset(inputs, targets) #.unsqueeze(2), .unsqueeze(1)

    return dataset

# --------------------------------------- #
# old functions no longer used 

# def create_pairs(train_val_test, reach, year_target=5, nodata_threshold=400000, water_threshold=12000, 
#                  dir_folders=r'data\satellite\dataset', collection=r'JRC_GSW1_4_MonthlyHistory'):
#     '''
#     This function pairs the input and target datasets depending on use and reach. It also applies an image selection based on no-data and water pixels threshold
#     (max and min, respectively) for which those couple that have one or more images (both inputs and targets) that do not resepct the given threshold will be discarded.  
    
#     Inputs:
#            train_val_test = str, specifies for what the images are used for.
#                             available options: 'training', 'validation' and 'testing'
#            reach = int, representing reach number. Number increases going upstream.
#                    default: 1, applies for both validation and testing.
#                    For training, the available range is 1-28 (included)
#            year_target = int, sets the year predicted after a sequence of input years.
#                          default: 5, input dataset is made of 4 images and 5th year is the predicted one
#            nodata_threshold = int, maximum amount of no-data pixels accepted for the training, validation and testing
#                               default: 400000, quite high compared to total number of pixels (500000) but best images were already filtered out 
#            water_threshold = int, minimum amount of water pixels accepted for the training, validation and testing
#                              default: 12000, based on pixels distribution 
#            dir_folders = str, directory where folders are stored
#                          default: r'data\satellite\dataset'
#            collection = str, specifies the satellite images collection.
#                         default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
    
#     Outputs:
#             pairs = dict, contains the paired input and target datasets as torch.tensor  
#     '''
#     input_dataset, target_dataset = create_datasets(train_val_test, reach, year_target, dir_folders, collection)

#     filtered_input_dataset = []
#     filtered_target_dataset = []

#     # Filtering pairs based on the specified threshold
#     for input_images, target_image in zip(input_dataset, target_dataset):
#         # check input images
#         input_nodata_thr = any(count_pixels(img, 0) > nodata_threshold for img in input_images)
#         input_water_thr = any(count_pixels(img, 2) < water_threshold for img in input_images)
#         # check target images
#         target_nodata_thr = count_pixels(target_image[0], 0) > nodata_threshold # select index 0 because target_image has shape (1, 1000, 500)
#         target_nodata_thr = count_pixels(target_image[0], 2) < water_threshold
#         # apply both conditions
#         if not input_nodata_thr and not input_water_thr and not target_nodata_thr and not target_nodata_thr:
#             filtered_input_dataset.append(input_images)
#             filtered_target_dataset.append(target_image)
        
#     # create tensors and pair these
#     input_tensors = [torch.stack([torch.tensor(img) for img in sequence]) for sequence in filtered_input_dataset]
#     target_tensors = [torch.tensor(target[0]) for target in filtered_target_dataset]
#     pairs = {'input': torch.stack(input_tensors), 'target': torch.stack(target_tensors)}
    
#     return pairs 

# def create_full_dataset(train_val_test, year_target=5, nodata_threshold=400000, water_threshold=12000, 
#                     dir_folders=r'data\satellite\dataset', collection=r'JRC_GSW1_4_MonthlyHistory'):
#     '''
#     This function generates the full dataset for the given use, combining all reaches. It is built upon the function `create_pairs`, 
#     which creates the paired couples input-target for a specific reach and given a specified value of no-data and water pixels thresholds.

#     It stacks all different pairs within one use in order to have the dataset ready for the training, validation and testing of the model.

#     Inputs:
#         train_val_test = str, specifies for what the images are used for.
#                             available options: 'training', 'validation' and 'testing'
#         year_target = int, sets the year predicted after a sequence of input years.
#                         default: 5, input dataset is made of 4 images and 5th year is the predicted one
#         nodata_threshold = int, maximum amount of no-data pixels accepted for the training, validation and testing
#                             default: 400000, quite high compared to total number of pixels (500000) but best images were already filtered out 
#         water_threshold = int, minimum amount of water pixels accepted for the training, validation and testing
#                             default: 12000, based on pixels distribution 
#         dir_folders = str, directory where folders are stored
#                         default: r'data\satellite\dataset'
#         collection = str, specifies the satellite images collection.
#                         default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
    
#     Output:
#         stacked_dict = dict, contains all coupled input-target samples for each reach and use stacked togther
#     '''
    
#     stacked_dict = {'input': [], 'target': []}
#     for folder in os.listdir(dir_folders):
#         if train_val_test in folder:
#             reach_id = folder.split('_r',1)[1]
#             pairs = create_pairs(train_val_test, int(reach_id), year_target, nodata_threshold, water_threshold, dir_folders, collection)
#             for key in pairs:
#                 stacked_dict[key].extend(pairs[key])
    
#     return stacked_dict