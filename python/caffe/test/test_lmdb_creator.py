import os
import unittest
import shutil
import logging

import lmdb
import numpy as np
import caffe.io
from caffe.proto.caffe_pb2 import Datum

from caffe.lmdb_creator import LMDBCreator

class TestCaffeLMDBCreator(unittest.TestCase):
    
    def setUp(self):
        logging_format = '[%(asctime)-19s, %(name)s] %(message)s'
        logging.basicConfig(level=logging.DEBUG,
                            format=logging_format)
        self.logger = logging.getLogger('UnitTest CaffeLMDBCreator')        
    
    def tearDown(self):
        pass

    def testBatchCreate(self):
        self.logger.info('Testing Batch Create')
        lmdb_path = os.path.join(os.path.dirname(__file__), 'test_lmdb')
        n_dummy_images = 10000
           
        dummy_data = [np.random.randint(0,256, (1, 224,224)).astype(np.uint8)
                      for _ in xrange(n_dummy_images)]
        labels = list(xrange(n_dummy_images))
           
        self.logger.info('Creating test LMDB')
        lmdb_creator = LMDBCreator()
        lmdb_creator.create_single_lmdb_from_ndarray_batch(array_list=dummy_data, labels_list=labels, 
                                                           lmdb_path=lmdb_path, max_lmdb_size=1024**3)
        self.logger.info('Testing previously created LMDB')
        database = lmdb.open(lmdb_path)
        self.assertEquals(first=database.stat()['entries'], second=n_dummy_images)
        with database.begin() as txn:
            cursor = txn.cursor()
            datum = Datum()
            for item_idx, (key, serialized_datum) in enumerate(cursor.iternext()):
                datum.ParseFromString(serialized_datum)
                mat = caffe.io.datum_to_array(datum=datum)
                # check if the key is correct
                self.assertEquals(first=key, second='%s_%d' % (str(item_idx).zfill(8), datum.label))
                # check if the ndarray is correct
                np.testing.assert_equal(actual=mat, desired=dummy_data[item_idx])
                   
        # clean up 
        database.close()        
        if os.path.exists(lmdb_path):
            shutil.rmtree(path=lmdb_path)
      
    def testSingleCreate(self):
        self.logger.info('Testing Single Create')
        lmdb_path = os.path.join(os.path.dirname(__file__), 'test_lmdb')
        n_dummy_images = 10000
          
        # create dummy data and dummy labels
        dummy_data = [np.random.randint(0,256, (1, 224,224)).astype(np.uint8)
                      for _ in xrange(n_dummy_images)]
        labels = np.array(('label1', 'label2', 'label3', 'label4'))
        label_map = dict((label, idx) for idx, label in enumerate(labels))
        inds = np.random.randint(0,4, n_dummy_images)        
        labels = labels[inds]
          
        self.logger.info('Creating test LMDB')
        lmdb_creator = LMDBCreator()
        lmdb_creator.open_single_lmdb_for_write(lmdb_path=lmdb_path, max_lmdb_size=1024**3, 
                                                create=True, label_map=label_map)
        for dummy_datum, label in zip(dummy_data, labels):
            lmdb_creator.put_single(img_mat=dummy_datum, label=str(label))
        lmdb_creator.finish_creation()
          
        self.logger.info('Testing previously created LMDB')
        # build the reverse label map
        idx_to_label_map = dict((v, k) for k,v in label_map.items())
        database = lmdb.open(lmdb_path)
        self.assertEquals(first=database.stat()['entries'], second=n_dummy_images)
        with database.begin() as txn:
            cursor = txn.cursor()
            datum = Datum()
            for item_idx, (key, serialized_datum) in enumerate(cursor.iternext()):
                datum.ParseFromString(serialized_datum)
                mat = caffe.io.datum_to_array(datum=datum)
                # check if the key is correct
                self.assertEquals(first=key, second='%s_%s' % (str(item_idx).zfill(8), idx_to_label_map[datum.label]))
                # check if the ndarray is correct
                np.testing.assert_equal(actual=mat, desired=dummy_data[item_idx])
                if (item_idx+1) % 1000 == 0 or (item_idx+1) == n_dummy_images:
                    self.logger.debug('   [ %*d / %d ] matrices passed test', len(str(n_dummy_images)), item_idx+1, n_dummy_images) 
                  
        # clean up
        database.close()
        if os.path.exists(lmdb_path):
            shutil.rmtree(path=lmdb_path)
    
    def testDualCreate(self):
        self.logger.info('Testing Dual Create')
        img_lmdb_path = os.path.join(os.path.dirname(__file__), 'img_test_lmdb')
        additional_lmdb_path = os.path.join(os.path.dirname(__file__), 'additional_test_lmdb')
        n_dummy_images = 9235
        
        # create dummy data and dummy labels
        img_dummy_data = [np.random.randint(0,256, (1,224,224)).astype(np.uint8)
                          for _ in xrange(n_dummy_images)]
        additional_dummy_data = [np.random.randint(0,256, (1,1,604)).astype(np.uint8)
                                 for _ in xrange(n_dummy_images)]
        labels = np.array(('label1', 'label2', 'label3', 'label4'))
        label_map = dict((label, idx) for idx, label in enumerate(labels))
        inds = np.random.randint(0,4, n_dummy_images)        
        labels = labels[inds]

        self.logger.info('Creating test LMDB')
        lmdb_creator = LMDBCreator()
        lmdb_creator.open_dual_lmdb_for_write(image_lmdb_path=img_lmdb_path, additional_lmdb_path=additional_lmdb_path, 
                                              max_lmdb_size=1024**3, create=True, label_map=label_map)
        for img_dummy_datum, additional_dummy_datum, label in zip(img_dummy_data,
                                                                  additional_dummy_data,
                                                                  labels):
            lmdb_creator.put_dual(img_mat=img_dummy_datum, additional_mat=additional_dummy_datum, label=str(label))
        lmdb_creator.finish_creation()
        
        self.logger.info('Testing previously created LMDB')
        # build the reverse label map
        idx_to_label_map = dict((v, k) for k,v in label_map.items())
        img_database = lmdb.open(img_lmdb_path)
        additional_database = lmdb.open(additional_lmdb_path)
        self.assertEquals(first=img_database.stat()['entries'], second=n_dummy_images)
        self.assertEquals(first=additional_database.stat()['entries'], second=n_dummy_images)
        with img_database.begin() as img_txn, additional_database.begin() as additional_txn:
            img_cursor = img_txn.cursor()
            additional_cursor = additional_txn.cursor()
            img_datum = Datum()
            additional_datum = Datum()
            for item_idx, ((img_key, img_serialized_datum), (additional_key, additional_serialized_datum)) in enumerate(zip(img_cursor.iternext(), additional_cursor.iternext())):
                img_datum.ParseFromString(img_serialized_datum)
                additional_datum.ParseFromString(additional_serialized_datum)
                img_mat = caffe.io.datum_to_array(img_datum)
                additional_mat = caffe.io.datum_to_array(additional_datum)
                # check the key
                self.assertEquals(first=img_key, second=additional_key)
                self.assertEquals(first=img_key, second='%s_%s' % (str(item_idx).zfill(8), idx_to_label_map[img_datum.label]))
                # check if the ndarray is correct
                np.testing.assert_equal(actual=img_mat, desired=img_dummy_data[item_idx])
                np.testing.assert_equal(actual=additional_mat, desired=additional_dummy_data[item_idx])
                if (item_idx+1) % 1000 == 0 or (item_idx+1) == n_dummy_images:
                    self.logger.debug('   [ %*d / %d ] matrices passed test', len(str(n_dummy_images)), item_idx+1, n_dummy_images)
        # clean up
        img_database.close()
        additional_database.close()
        if os.path.exists(img_lmdb_path):
            shutil.rmtree(path=img_lmdb_path)
        if os.path.exists(additional_lmdb_path):
            shutil.rmtree(path=additional_lmdb_path)
    
    def testShuffle(self):
        self.logger.info('Testing Data Shuffling')
        img_lmdb_path = os.path.join(os.path.dirname(__file__), 'img_test_lmdb')
        additional_lmdb_path = os.path.join(os.path.dirname(__file__), 'additional_test_lmdb')
        n_dummy_images = 9235
        
        # create dummy data and dummy labels
        img_dummy_data = [np.random.randint(0,256, (1,224,224)).astype(np.uint8)
                          for _ in xrange(n_dummy_images)]
        additional_dummy_data = [np.random.randint(0,256, (1,1,604)).astype(np.uint8)
                                 for _ in xrange(n_dummy_images)]
        labels = np.array(('label1', 'label2', 'label3', 'label4'))
        label_map = dict((label, idx) for idx, label in enumerate(labels))
        inds = np.random.randint(0,4, n_dummy_images)        
        labels = labels[inds]
        
        # create random order to insert
        rand_indices = np.arange(n_dummy_images)
        np.random.shuffle(rand_indices)
        
        # create the LMDB
        self.logger.info('Creating test LMDB')
        lmdb_creator = LMDBCreator()
        lmdb_creator.open_dual_lmdb_for_write(image_lmdb_path=img_lmdb_path, additional_lmdb_path=additional_lmdb_path, 
                                              max_lmdb_size=1024**3, create=True, label_map=label_map)
        for idx, (img_dummy_datum, additional_dummy_datum, label) in enumerate(zip(img_dummy_data,
                                                                                   additional_dummy_data,
                                                                                   labels)):
            lmdb_creator.put_dual(img_mat=img_dummy_datum, additional_mat=additional_dummy_datum, 
                                  label=str(label), key='%s_%s' % (str(rand_indices[idx]).zfill(8), str(label)))
        lmdb_creator.finish_creation()
            
        self.logger.info('Testing previously created LMDB')
        # build the reverse label map
        idx_to_label_map = dict((v, k) for k,v in label_map.items())
        img_database = lmdb.open(img_lmdb_path)
        additional_database = lmdb.open(additional_lmdb_path)
        self.assertEquals(first=img_database.stat()['entries'], second=n_dummy_images)
        self.assertEquals(first=additional_database.stat()['entries'], second=n_dummy_images)
        with img_database.begin() as img_txn, additional_database.begin() as additional_txn:
            img_cursor = img_txn.cursor()
            additional_cursor = additional_txn.cursor()
            img_datum = Datum()
            additional_datum = Datum()
            for item_idx, ((img_key, img_serialized_datum), (additional_key, additional_serialized_datum)) in enumerate(zip(img_cursor.iternext(), additional_cursor.iternext())):
                img_datum.ParseFromString(img_serialized_datum)
                additional_datum.ParseFromString(additional_serialized_datum)
                img_mat = caffe.io.datum_to_array(img_datum)
                additional_mat = caffe.io.datum_to_array(additional_datum)
                # check the key
                self.assertEquals(first=img_key, second=additional_key)
                self.assertEquals(first=img_key, second='%s_%s' % (str(item_idx).zfill(8), idx_to_label_map[img_datum.label]))
                # check if the ndarray is correct
                np.testing.assert_equal(actual=img_mat, desired=img_dummy_data[np.where(rand_indices == item_idx)[0][0]])
                np.testing.assert_equal(actual=additional_mat, desired=additional_dummy_data[np.where(rand_indices == item_idx)[0][0]])
                if (item_idx+1) % 1000 == 0 or (item_idx+1) == n_dummy_images:
                    self.logger.debug('   [ %*d / %d ] matrices passed test', len(str(n_dummy_images)), item_idx+1, n_dummy_images)
        # clean up
        img_database.close()
        additional_database.close()
        if os.path.exists(img_lmdb_path):
            shutil.rmtree(path=img_lmdb_path)
        if os.path.exists(additional_lmdb_path):
            shutil.rmtree(path=additional_lmdb_path)
            
                
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testCreateCaffeLMDB']
    unittest.main()
