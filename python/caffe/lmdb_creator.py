import os
import shutil
import logging

import numpy as np
import lmdb
import caffe.io

class LMDBCreator(object):
    def __init__(self):
        '''
        The LMDB creator class can create a single LMDB for single label 
        classification or two LMDBs where each element in the database_images 
        has a corresponding counterpart in database_additional with the same 
        key. This is useful for creating e.g. LMDBs for semantic-segmentation.
        '''
        self.logger = logging.getLogger('LMDBCreator')
        self.database_images = None
        self.database_additional = None
        self.txn_images = None
        self.txn_additional = None
        self.label_map = None
        self.internal_counter = 0
        
        self.logger.debug('Using LMDB version %d.%d.%d' % lmdb.version())
    
    def open_single_lmdb_for_write(self, lmdb_path, max_lmdb_size=1024**4, 
				   create=True, label_map=None):
        '''
        Opens a single LMDB for inserting ndarrays (i.e. images)
        @param lmdb_path: str
            Where to save the LMDB
        @param max_lmdb_size: int
            The maximum size in bytes of the LMDB (default: 1TB)
        @param create: bool
            If this flag is set, a potentially previously created LMDB at 
            lmdb_path is deleted and overwritten by this new LMDB
        @param label_map: dictionary
            If you supply a dictionary mapping string labels to integer indices
            you can later call put_single with string labels instead of int 
            labels
        '''
        # delete existing LMDB if necessary
        if os.path.exists(lmdb_path) and create:
            self.logger.info('Erasing LMDB at %s', lmdb_path)
            shutil.rmtree(lmdb_path)
        self.logger.info('Opening single LMDB at %s for writing', lmdb_path)
        self.database_images = lmdb.open(path=lmdb_path, 
					 map_size=max_lmdb_size)
        self.txn_images = self.database_images.begin(write=True)
        self.label_map = label_map
    
    def open_dual_lmdb_for_write(self, image_lmdb_path, additional_lmdb_path, 
				 max_lmdb_size=1024**4, create=True, label_map=None):
        '''
        Opens two LMDBs for writing where each element in the first has a 
        counterpart with the same key in the second 
        @param image_lmdb_path: str
            Where to save the image LMDB
        @param additional_lmdb_path: str
            Where to save the additional LMDB
        @param max_lmdb_size: int
            The maximum size in bytes of each LMDB (default: 1TB)
        @param create: bool
            If this flag is set, potentially previously created LMDBs at 
            lmdb_path and additional_lmdb_path are deleted and overwritten by 
            new LMDBs
        @param label_map: dictionary
            If you supply a dictionary mapping string labels to integer indices
            you can later call put_dual with string labels instead of int 
            labels
        '''
        # delete existing LMDBs if necessary
        if os.path.exists(image_lmdb_path) and create:
            self.logger.info('Erasing LMDB at %s', image_lmdb_path)
            shutil.rmtree(image_lmdb_path)
        if os.path.exists(additional_lmdb_path) and create:
            self.logger.info('Erasing LMDB at %s', additional_lmdb_path)
            shutil.rmtree(additional_lmdb_path)            
        self.logger.info('Opening LMDBs at %s and %s for writing', 
			 image_lmdb_path, additional_lmdb_path)
        self.database_images = lmdb.open(path=image_lmdb_path, 
					 map_size=max_lmdb_size)
        self.txn_images = self.database_images.begin(write=True)
        self.database_additional = lmdb.open(path=additional_lmdb_path, 
					     map_size=max_lmdb_size)
        self.txn_additional = self.database_additional.begin(write=True)
        self.label_map = label_map
    
    def put_single(self, img_mat, label, key=None):
        '''
        Puts an ndarray into the previously opened single LMDB        
        @param img_mat: ndarray
            The image data to be inserted in the LMDB
        @param label: str or int
            The label for the image
        @param key: str
            The key under which to save the data in the LMDB
            If key is None, a generic key is generated
        '''
        # some checks
        if self.database_images is None:
            raise ValueError('No LMDB to write to. \
	      Did you call open_single_lmdb_for_write?')
        if self.database_additional is not None:
            raise ValueError('Cannot execute put_single as \
	      open_dual_lmdb_for_write has been chosen for LMDB creation')
        if img_mat.dtype != np.uint8 or img_mat.ndim != 3:
            raise ValueError('img_mat must be a 3d-ndarray of type np.uint8')
        
        # label may be a string if a label map was supplied
        datum_label = None
        if type(label) == str: 
            if self.label_map is None:
                raise ValueError('You may only supply a label of type str if \
		  you called open_single_lmdb_for_write with a valid label_map')
            else:
                datum_label = self.label_map[label]
        elif type(label) == int:
            datum_label = label
        else:
            raise ValueError('label must be of type str or int')
        
        # convert img_mat to Caffe Datum
        datum = caffe.io.array_to_datum(arr=img_mat, label=datum_label)
        if key is None:
            key = '%s_%s' % (str(self.internal_counter).zfill(8), str(label))
        # push Datum into the LMDB
        self.txn_images.put(key=key, value=datum.SerializeToString())
        self.internal_counter += 1
        if self.internal_counter % 1000 == 0:
            self.txn_images.commit()
            self.logger.debug('   Finished %*d ndarrays', 
			      8, self.internal_counter)
            # after a commit the txn object becomes invalid, so we need to get 
            # a new one
            self.txn_images = self.database_images.begin(write=True)
    
    def put_dual(self, img_mat, additional_mat, label, key=None):
        '''
        Puts an image and its corresponding additional information ndarray into
        the previously opened LMDBs
        
        @param img_mat: ndarray
            The image data to be inserted in the LMDB
        @param additional_mat: ndarray
            The label matrix (attributes, PHOC, ...) to be inserted
        @param label: str or int
            The label for the image
        @param key: str
            The key under which to save the data in the LMDB
            If key is None, a generic key is generated
        '''
        # some checks
        if self.database_images is None:
            raise ValueError('No LMDB to write to. \
	      Did you call open_dual_lmdb_for_write?')
        if self.database_additional is None:
            raise ValueError('Cannot execute put_dual as \
	      open_single_lmdb_for_write has been chosen for LMDB creation')
        if img_mat.dtype != np.uint8 or img_mat.ndim != 3:
            raise ValueError('img_mat must be a 3d-ndarray of type np.uint8')
        if additional_mat.dtype != np.uint8 or additional_mat.ndim != 3:
            raise ValueError('additional_mat must be a 3d-ndarray of \
	      type np.uint8')
        
        # label may be a string if a label map was supplied
        datum_label = None
        if type(label) == str: 
            if self.label_map is None:
                raise ValueError('You may only supply a label of type str if \
		  you called open_dual_lmdb_for_write with a valid label_map')
            elif not label in self.label_map.keys():
                self.logger.warn('Warning, unknown key - skipping this entry')
                return
            else:
                datum_label = self.label_map[label]
        elif type(label) == int:
            datum_label = label
        else:
            raise ValueError('label must be of type str or int')
        
        # convert img_mat and additional_mat to Caffe Data
        datum_img = caffe.io.array_to_datum(arr=img_mat, label=datum_label)
        datum_additional = caffe.io.array_to_datum(arr=additional_mat, 
						   label=datum_label)
        if key is None:
            key = '%s_%s' % (str(self.internal_counter).zfill(8), str(label))
        # push Data in the current LMDBs 
        self.txn_images.put(key=key, value=datum_img.SerializeToString())
        self.txn_additional.put(key=key, 
				value=datum_additional.SerializeToString())
        self.internal_counter += 1
        if self.internal_counter % 10000 == 0:
            self.txn_images.commit()
            self.txn_additional.commit()
            self.logger.debug('   Finished %*d ndarrays', 
			      8, self.internal_counter)
            # after a commit the txn objects becomes invalid, so we need to get
            # new ones
            self.txn_images = self.database_images.begin(write=True)
            self.txn_additional = self.database_additional.begin(write=True)
        return
    
    def finish_creation(self):
        '''
        Wraps up LMDB creation and resets all internal variables
        '''
        self.txn_images.commit()
        self.database_images.sync()
        self.database_images.close()
        if self.database_additional is not None:
            self.txn_additional.commit()
            self.database_additional.sync()
            self.database_additional.close()
        self.logger.info('Finished after writing %d ndarrays', 
			 self.internal_counter)
        self.database_images = None
        self.database_additional = None
        self.txn_images = None
        self.txn_additional = None
        self.label_map = None
        self.internal_counter = 0
    
    def create_single_lmdb_from_ndarray_batch(self, array_list, labels_list, 
					      lmdb_path, max_lmdb_size=1024**4, 
					      create=True):
        '''
        Creates an LMDB from a list of ndarrays
        @param mats: list of 3d-ndarrays
        @param labels: list of str or int
        @param lmdb_path: str
        '''
        # some checks
        if type(labels_list[0]) != str and type(labels_list[0]) != int:
            raise ValueError('Labels must be of type string or int32')
        if len(labels_list) != len(array_list):
            raise ValueError('Number of labels and number of ndarrays must \
	      match')
        if array_list[0].dtype != np.uint8 or array_list[0].ndim != 3:
            raise ValueError('array_list must contain 3d-ndarray of type \
	      np.uint8')
        
        # delete previously created database
        if create and os.path.exists(lmdb_path):
            self.logger.info('Removing previously generated LMDB at %s', 
			     lmdb_path)
            shutil.rmtree(path=lmdb_path)
        
        self.logger.info('Creating LMDB from list of ndarrays')
        label_to_idx_dict = None
        if type(labels_list[0]) == str:
            self.logger.info('Found labels to be strings, creating label \
	      map...')
            unique_labels = sorted(list(set(labels_list)))
            label_to_idx_dict = dict((label, idx) 
				     for idx, label in enumerate(unique_labels))
            labels_list = [label_to_idx_dict[label] for label in labels_list]
        
        # open LMDB
        self.logger.info('Writing LMDB...')
        self.database = lmdb.open(lmdb_path, map_size=max_lmdb_size)
        txn = self.database.begin(write=True)
        # put the datums in the LMDB
        for mat_idx, (mat, label) in enumerate(zip(array_list, labels_list)):
            datum = caffe.io.array_to_datum(arr=mat, label=label)
            key = '%s_%s' % (str(mat_idx).zfill(8), label)
            txn.put(key=key, value=datum.SerializeToString())
            # write every 1000 transactions
            if (mat_idx+1) % 1000 == 0 or (mat_idx+1) == len(array_list):                    
                txn.commit()
                txn = self.database.begin(write=True)
                self.logger.debug('   [ %*d / %d ]', len(str(len(array_list))), 
				  mat_idx+1, len(array_list))
	# finish LMDB creation
        self.database.sync()
        self.database.close()
        
        self.logger.info('Wrote %s ndarrays to the LMDB at %s', 
			 len(array_list), lmdb_path)
                
