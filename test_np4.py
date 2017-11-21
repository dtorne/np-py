#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:03:33 2017

@author: david
"""

from np1 import *
import unittest
import testbase

 
def wordCount(rdd):
    wcntRdd = rdd.flatMap(lambda line: line.split()).\
        map(lambda word: (word, 1)).\
        reduceByKey(lambda fa, fb: fa + fb)
    return wcntRdd
 
class TestWordCount(testbase.ReusedPySparkTestCase):
    def test_word_count(self):
        rdd = self.sc.parallelize(["a b c d", "a c d e", "a d e f"])
        res = wordCount(rdd)
        res = res.collectAsMap()
        expected = {"a":3, "b":1, "c":2, "d":3, "e":2, "f":1}
        self.assertEqual(res,expected)


class TestMethods(unittest.TestCase):

    def setUp(self):
        self.path = "uf20-01.cnf"
        (self.numVar, self.numClauses, self.clauses) = parseNPFunction(path = self.path)

       
    def test_read_file(self):
        
        # Test parses the 0 and reads clauses and variables
        test = self.clauses.pop()
        self.assertTrue(len(test.intersection(set([0]))) == 0)
        clauses.add(test)
        self.assertTrue(self.numClauses > self.numVar)
        
    def test_min_clause(self):
        
        currentLevel = 18    
        minClauses = np1.minimClause(self.clauses, currentLevel, self.newVar)
        #Includes the newVar in all clauses
        hasNewVariables = True
        #Has newVar positive clauses
        hasPosSign = False
        #Has newVar negative clauses
        hasNegSign = False
        for clause in minClauses:
            if newVar not in clause and -newVar not in clause:
                hasNewVariables = False
            if newVar in clause:
                hasPosSign = True
            elif -newVar in clause:
                hasNegSign = True
                
        self.assertTrue(hasNewVariables)
        self.assertTrue(hasPosSign and hasNegSign)
        
        minClauses = np1.minimClause(clauses, currentLevel, self.newVar, False)
        for clause in minClauses:
            if newVar not in clause and -newVar not in clause:
                hasNewVariables = False
            if newVar in clause:
                hasPosSign = True
            elif -newVar in clause:
                hasNegSign = True
                
        self.assertTrue(not hasNewVariables)
        self.assertTrue(hasPosSign and hasNegSign)

        minClause = np1.minimClause(clauses,level=8,newVariable=9)

    def test_checkBoolean(self):
        currentLevel = 18    
        minClauses = np1.minimClause(self.clauses, currentLevel, self.newVar)
        #Based on minClause = 
        #{frozenset({9, -5, -1}),
        # frozenset({-8, 4, 7}), 
        #frozenset({9, 6, 1}),
        # frozenset({-8, 4, -9}),
        # frozenset({1, -5, -9})}

        instanceSet = {9,-5,-1}
        passes = checkBoolean(self.clauses, instanceSet)
        self.assertTrue(not passes)

        instanceSet = {1,5,-9}
        passes = checkBoolean(self.clauses, instanceSet)
        self.assertTrue(passes)
    
    def test_update_state(self):
        currentLevel = 18    
        minClauses = np1.minimClause(self.clauses, currentLevel, self.newVar)
        
        #Based on minClause = 
        #{frozenset({9, -5, -1}),
        # frozenset({-8, 4, 7}), 
        #frozenset({9, 6, 1}),
        # frozenset({-8, 4, -9}),
        # frozenset({1, -5, -9})}
        #Value 100000001 only var1 at 1 and newVar=9 positive is not blocked by any
        #Value 000000001 all negative included newVar=-9 is blocked by {1, -5, -9}
        # So status should be 10b which is 2
        status2 = np1.updateState(clauses, idValue=1, status=3, level=8, newVariable=9)
        self.assertTrue(status2==2)
        #Value 100000000 only newVar=9 positive is blocked by {9, -5, -1}
        #Value 000000000 all negative included newVar=-9 is not blocked by any
        # So status should be 01b which is 1
        status1 = np1.updateState(clauses, idValue=0, status=3, level=8, newVariable=9)
        self.assertTrue(status1==1)


if __name__ == '__main__':
    if __name__ == '__main__':
        unittest.main()