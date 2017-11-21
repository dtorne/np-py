import os
import sys
import unittest
 
 
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
 
sc_values = {}
 
class ReusedPySparkTestCase(unittest.TestCase):
 
    @classmethod
    def setUpClass(cls):
        conf = SparkConf().setMaster("local[2]") \
            .setAppName(cls.__name__) \
        cls.sc = SparkContext(conf=conf)
        sc_values[cls.__name__] = cls.sc
 
    @classmethod
    def tearDownClass(cls):
        print "....calling stop tearDownClas, the content of sc_values=", sc_values
        sc_values.clear()
        cls.sc.stop()
 
class PySparkTestCase(unittest.TestCase):
 
    def setUp(self):
        self._old_sys_path = list(sys.path)
        conf = SparkConf().setMaster("local[2]") \
            .setAppName(self.__class__.__name__) \
        self.sc = SparkContext(conf=conf)
 
    def tearDown(self):
        self.sc.stop()
        sys.path = self._old_sys_path
 
class TestResusedScA(ReusedPySparkTestCase):
 
    def testA_1(self):
        rdd = self.sc.parallelize([1,2,3])
        self.assertEqual(rdd.collect(), [1,2,3])
        sc_values['testA_1'] = self.sc
 
    def testA_2(self):
        sc_values['testA_2'] = self.sc
 
        self.assertEquals(self.sc, sc_values['testA_1'])
 
 
class TestResusedScB(ReusedPySparkTestCase):
    def testB_1(self):
        sc_values['testB_1'] = self.sc
 
    def testB_2(self):
        sc_values['testB_2'] = self.sc
 
    def testB_3(self):
        sc_values['testB_3'] = self.sc
        self.assertEquals(self.sc, sc_values['testB_2'])
 
 
if __name__ == '__main__':
    unittest.main()