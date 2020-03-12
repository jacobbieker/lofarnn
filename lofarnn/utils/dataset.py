import os


def mkdirs_safe(directory_list):
    """When given a list containing directories,
    checks if these exist, if not creates them."""
    assert isinstance(directory_list, list)
    for directory in directory_list:
        os.makedirs(directory, exist_ok=True)

def create_recursive_directories(prepend_path, current_dir, dictionary):
    """Give it a base, a current_dir to create and a dictionary of stuff yet to create.
    See create_VOC_style_directory_structure for a use case scenario."""
    mkdirs_safe([os.path.join(prepend_path,current_dir)])
    if dictionary[current_dir] == None:
        return
    else:
        for key in dictionary[current_dir].keys():
            create_recursive_directories(os.path.join(prepend_path, current_dir), key,
                                         dictionary[current_dir])

def create_COCO_style_directory_structure(root_directory, suffix=''):
    """
     We will create a directory structure identical to that of the CLARAN
     The root directory is the directory in which the directory named 'RGZdevkit' will be placed.
     The structure contained will be as follows:
     LGZ_COCOstyle{suffix}/
        |-- Annotations/
             |-- *.json (Annotation files)
        |-- all/ (train,test,val split directory)
             |-- *.png (Image files)
        |-- train/ (train,test,val split directory)
             |-- *.png (Image files)
        |-- val/ (train,test,val split directory)
             |-- *.png (Image files)
        |-- test/ (train,test,val split directory)
             |-- *.png (Image files)
    """
    directories_to_make = {f'LGZ_COCOstyle{suffix}':
                               {'annotations':None,
                                'all':None,
                                'train':None,
                                'val':None,
                                'test':None }}
    create_recursive_directories(root_directory,f'LGZ_COCOstyle{suffix}' ,directories_to_make)
    print(f'COCO style directory structure created in \'{root_directory}\'.\n')
    all_directory, train_directory, val_directory, test_directory, annotations_directory = \
        os.path.join(root_directory,f'LGZ_COCOstyle{suffix}', 'all'), \
        os.path.join(root_directory,f'LGZ_COCOstyle{suffix}', 'train'), \
        os.path.join(root_directory,f'LGZ_COCOstyle{suffix}','val'), \
        os.path.join(root_directory,f'LGZ_COCOstyle{suffix}','test'), \
        os.path.join(root_directory,f'LGZ_COCOstyle{suffix}','annotations')
    return all_directory, train_directory, val_directory, test_directory, annotations_directory