import numpy as np
from tqdm import tqdm
import csv
import inspect
import os
import sys
import warnings
from collections import defaultdict
from copy import deepcopy
from PIL import Image
def parse_german_traffic_sign(train_dir,
                              test_dir,
                              test_annotations,
                              input_format,
                              reshape=None,
                              ret=True,
                              verbose=True):
    X_train,Y_train = parse_csv_distributed(train_dir,input_format,reshape=reshape,ret=True)
    X_test,Y_test = parse_csv_traffic_signs(test_dir,test_annotations,input_format,reshape=reshape,ret=True)
    
    return (X_train,Y_train),(X_test,Y_test)

def parse_csv_distributed(images_dir,
                          input_format,
                          include_classes='all',
                          reshape=None,
                          ret=True,
                          verbose=True):
    '''
    Arguments:
        images_dir (str): The path to the directory that contains the images.
        labels_filename (str): The filepath to a CSV file that contains one ground truth bounding box per line
            and each line contains the following six items: image file name, class ID, xmin, xmax, ymin, ymax.
            The six items do not have to be in a specific order, but they must be the first six columns of
            each line. The order of these items in the CSV file must be specified in `input_format`.
            The class ID is an integer greater than zero. Class ID 0 is reserved for the background class.
            `xmin` and `xmax` are the left-most and right-most absolute horizontal coordinates of the box,
            `ymin` and `ymax` are the top-most and bottom-most absolute vertical coordinates of the box.
            The image name is expected to be just the name of the image file without the directory path
            at which the image is located.
        input_format (list): A list of six strings representing the order of the six items
            image file name, class ID, xmin, xmax, ymin, ymax in the input CSV file. The expected strings
            are 'image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'.
        include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
            are to be included in the dataset. If 'all', all ground truth boxes will be included in the dataset.
        random_sample (float, optional): Either `False` or a float in `[0,1]`. If this is `False`, the
            full dataset will be used by the generator. If this is a float in `[0,1]`, a randomly sampled
            fraction of the dataset will be used, where `random_sample` is the fraction of the dataset
            to be used. For example, if `random_sample = 0.2`, 20 precent of the dataset will be randomly selected,
            the rest will be ommitted. The fraction refers to the number of images, not to the number
            of boxes, i.e. each image that will be added to the dataset will always be added with all
            of its boxes.
        ret (bool, optional): Whether or not to return the outputs of the parser.
        verbose (bool, optional): If `True`, prints out the progress for operations that may take a bit longer.

    Returns:
        None by default, optionally lists for whichever are available of images, image filenames, labels, and image IDs.
    '''

    # Set class members.
    images_dir = images_dir
    load_images_into_memory=True
    input_format = input_format
    include_classes = include_classes
    labels_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax']
    # Before we begin, make sure that we have a labels_filename and an input_format

    # Erase data that might have been parsed before
    filenames = []
    image_ids = []
    labels = []

    # First, just read in the CSV file lines and sort them.
    data = []
    print(os.getcwd())
    for c in tqdm(range(0, 43)):
        prefix = images_dir + '/' + format(c, '05d') + '/'  # subdirectory for class
        labels_filename = prefix + 'GT-' + format(c, '05d') + '.csv'  # annotations file
        with open(labels_filename, newline='') as csvfile:
            csvread = csv.reader(csvfile, delimiter=';')
            next(csvread)  # Skip the header row.
            for row in csvread:  # For every line (i.e for every bounding box) in the CSV file...
                if include_classes == 'all':  # If the class_id is among the classes that are to be included in the dataset...
                    box = []  # Store the box class and coordinates here
                    box.append(format(c, '05d') + '/' + row[input_format.index(
                        'Filename')].strip())  # Select the image name column in the input format and append its content to `box`
                    #item_dict = {'image_name': row[input_format.index('Filename')].strip(),
                    #             'class_id': int(row[input_format.index('ClassId')].strip()),
                    #             'xmin': int(row[input_format.index('Roi.X1')].strip()),
                    #             'ymin': int(row[input_format.index('Roi.Y1')].strip()),
                    #             'xmax': int(row[input_format.index('Roi.X2')].strip()),
                    #             'ymax': int(row[input_format.index('Roi.Y2')].strip())}
                    item_dict = {'class_id':int(row[input_format.index('ClassId')].strip())}
                    #for element in labels_output_format:  # For each element in the output format (where the elements are the class ID and the four box coordinates)...
                    #    box.append(item_dict[element])
                    box.append(item_dict['class_id'])
                    data.append(box)

    data = sorted(data)  # The data needs to be sorted, otherwise the next step won't give the correct result

    # Now that we've made sure that the data is sorted by file names,
    # we can compile the actual samples and labels lists

    current_file = data[0][0]  # The current image for which we're collecting the ground truth boxes
    current_image_id = data[0][0].split('.')[
        0]  # The image ID will be the portion of the image name before the first dot.
    current_labels = []  # The list where we collect all ground truth boxes for a given image
    add_to_dataset = False
    for i, box in enumerate(data):
        if box[0] == current_file:  # If this box (i.e. this line of the CSV file) belongs to the current image file
            current_labels.append(box[1:])
            if i == len(data) - 1:  # If this is the last line of the CSV file
                if random_sample:  # In case we're not using the full dataset, but a random sample of it.
                    p = np.random.uniform(0, 1)
                    if p >= (1 - random_sample):
                        labels.append(np.stack(current_labels, axis=0))
                        filenames.append(os.path.join(images_dir, current_file))
                        image_ids.append(current_image_id)
                else:
                    labels.append(np.stack(current_labels, axis=0))
                    filenames.append(os.path.join(images_dir, current_file))
                    image_ids.append(current_image_id)
        else:  # If this box belongs to a new image file
            labels.append(np.stack(current_labels, axis=0))
            filenames.append(os.path.join(images_dir, current_file))
            image_ids.append(current_image_id)
            current_labels = []  # Reset the labels list because this is a new file.
            current_file = box[0]
            current_image_id = box[0].split('.')[0]
            current_labels.append(box[1:])
            if i == len(data) - 1:  # If this is the last line of the CSV file
                labels.append(np.stack(current_labels, axis=0))
                filenames.append(os.path.join(images_dir, current_file))
                image_ids.append(current_image_id)

    dataset_size = len(filenames)
    dataset_indices = np.arange(dataset_size, dtype=np.int32)
    lab = []
    for label in labels:
        lab.append(label[0])
    labels = lab
    if load_images_into_memory:
        images = []
        if verbose:
            it = tqdm(filenames, desc='Loading images into memory', file=sys.stdout)
        else:
            it = filenames
        for filename in it:
            with Image.open(filename) as image:
                if reshape is not None:
                    image = image.resize(size= reshape)
                    images.append(np.array(image, dtype=np.uint8))
                else:
                    images.append(np.array(image, dtype=np.uint8))
                
        if ret:  # In case we want to return these
            images = np.asarray(images)
            labels = np.asarray(labels)
            return images,labels

            
def parse_csv_traffic_signs(images_dir,
                            labels_filename,
                            input_format,
                            include_classes='all',
                            split='train',
                            random_sample=False,
                            reshape=None,
                            ret=True,
                            verbose=True,
                            reset_variables=True):
    '''
    Arguments:
        images_dir (str): The path to the directory that contains the images.
        labels_filename (str): The filepath to a CSV file that contains one ground truth bounding box per line
            and each line contains the following six items: image file name, class ID, xmin, xmax, ymin, ymax.
            The six items do not have to be in a specific order, but they must be the first six columns of
            each line. The order of these items in the CSV file must be specified in `input_format`.
            The class ID is an integer greater than zero. Class ID 0 is reserved for the background class.
            `xmin` and `xmax` are the left-most and right-most absolute horizontal coordinates of the box,
            `ymin` and `ymax` are the top-most and bottom-most absolute vertical coordinates of the box.
            The image name is expected to be just the name of the image file without the directory path
            at which the image is located.
        input_format (list): A list of six strings representing the order of the six items
            image file name, class ID, xmin, xmax, ymin, ymax in the input CSV file. The expected strings
            are 'image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'.
        include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
            are to be included in the dataset. If 'all', all ground truth boxes will be included in the dataset.
        random_sample (float, optional): Either `False` or a float in `[0,1]`. If this is `False`, the
            full dataset will be used by the generator. If this is a float in `[0,1]`, a randomly sampled
            fraction of the dataset will be used, where `random_sample` is the fraction of the dataset
            to be used. For example, if `random_sample = 0.2`, 20 precent of the dataset will be randomly selected,
            the rest will be ommitted. The fraction refers to the number of images, not to the number
            of boxes, i.e. each image that will be added to the dataset will always be added with all
            of its boxes.
        ret (bool, optional): Whether or not to return the outputs of the parser.
        verbose (bool, optional): If `True`, prints out the progress for operations that may take a bit longer.

    Returns:
        None by default, optionally lists for whichever are available of images, image filenames, labels, and image IDs.
    '''

    
    # Set class members.
    images_dir = images_dir
    load_images_into_memory=True
    input_format = input_format
    include_classes = include_classes
    labels_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax']
    # Before we begin, make sure that we have a labels_filename and an input_format
    if labels_filename is None or input_format is None:
        raise ValueError(
            "`labels_filename` and/or `input_format` have not been set yet. You need to pass them as arguments.")

    if reset_variables:
        # Erase data that might have been parsed before
        filenames = []
        image_ids = []
        labels = []
        highest_class_id = 0

        # First, just read in the CSV file lines and sort them.

    data = []

    with open(labels_filename, newline='') as csvfile:
        csvread = csv.reader(csvfile, delimiter=';')
        next(csvread)  # Skip the header row.
        # row_count = sum(1 for r in csvread)
        for row in csvread:  # For every line (i.e for every bounding box) in the CSV file...
            if include_classes == 'all':  # If the class_id is among the classes that are to be included in the dataset...
                box = []  # Store the box class and coordinates here
                box.append(row[input_format.index(
                    'Filename')].strip())  # Select the image name column in the input format and append its content to `box`
                item_dict = {#'image_name': row[input_format.index('Filename')].strip(),
                             'class_id': 1 + highest_class_id + int(
                                 row[input_format.index('ClassId')].strip())}
                             #'xmin': int(row[input_format.index('Roi.X1')].strip()),
                             #'ymin': int(row[input_format.index('Roi.Y1')].strip()),
                             #'xmax': int(row[input_format.index('Roi.X2')].strip()),
                             #'ymax': int(row[input_format.index('Roi.Y2')].strip())}
                #for element in labels_output_format:  # For each element in the output format (where the elements are the class ID and the four box coordinates)...
                box.append(item_dict['class_id'])
                data.append(box)

    data = sorted(data)  # The data needs to be sorted, otherwise the next step won't give the correct result

    # Now that we've made sure that the data is sorted by file names,
    # we can compile the actual samples and labels lists

    current_file = data[0][0]  # The current image for which we're collecting the ground truth boxes
    current_image_id = data[0][0].split('.')[
        0]  # The image ID will be the portion of the image name before the first dot.
    current_labels = []  # The list where we collect all ground truth boxes for a given image
    add_to_dataset = False
    for i, box in enumerate(data):

        if box[0] == current_file:  # If this box (i.e. this line of the CSV file) belongs to the current image file
            current_labels.append(box[1:])
            if i == len(data) - 1:  # If this is the last line of the CSV file
                if random_sample:  # In case we're not using the full dataset, but a random sample of it.
                    p = np.random.uniform(0, 1)
                    if p >= (1 - random_sample):
                        labels.append(np.stack(current_labels, axis=0))
                        filenames.append(os.path.join(images_dir, current_file))
                        image_ids.append(current_image_id)
                else:
                    labels.append(np.stack(current_labels, axis=0))
                    filenames.append(os.path.join(images_dir, current_file))
                    image_ids.append(current_image_id)
        else:  # If this box belongs to a new image file
            if random_sample:  # In case we're not using the full dataset, but a random sample of it.
                p = np.random.uniform(0, 1)
                if p >= (1 - random_sample):
                    labels.append(np.stack(current_labels, axis=0))
                    filenames.append(os.path.join(images_dir, current_file))
                    image_ids.append(current_image_id)
            else:
                labels.append(np.stack(current_labels, axis=0))
                filenames.append(os.path.join(images_dir, current_file))
                image_ids.append(current_image_id)
            current_labels = []  # Reset the labels list because this is a new file.
            current_file = box[0]
            current_image_id = box[0].split('.')[0]
            current_labels.append(box[1:])
            if i == len(data) - 1:  # If this is the last line of the CSV file
                if random_sample:  # In case we're not using the full dataset, but a random sample of it.
                    p = np.random.uniform(0, 1)
                    if p >= (1 - random_sample):
                        labels.append(np.stack(current_labels, axis=0))
                        filenames.append(os.path.join(images_dir, current_file))
                        image_ids.append(current_image_id)
                else:
                    labels.append(np.stack(current_labels, axis=0))
                    filenames.append(os.path.join(images_dir, current_file))
                    image_ids.append(current_image_id)

    dataset_size = len(filenames)
    dataset_indices = np.arange(dataset_size, dtype=np.int32)
    lab = []
    for label in labels:
        lab.append(label[0])
    labels = lab
    if load_images_into_memory:
        images = []
        if verbose:
            it = tqdm(filenames, desc='Loading images into memory', file=sys.stdout)
        else:
            it = filenames
        for filename in it:
            with Image.open(filename) as image:
                if reshape is not None:
                    image = image.resize(size= reshape)
                    images.append(np.array(image, dtype=np.uint8))
                else:
                    images.append(np.array(image, dtype=np.uint8))
                
        if ret:  # In case we want to return these
            images = np.asarray(images)
            labels = np.asarray(labels)
            return images,labels
