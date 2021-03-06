#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:03:33 2017

@author: david
"""

from np import *
import unittest
newVar = 19


class TestBookMethods(unittest.TestCase):

    def setUp(self):
        
        self.npSolver = NPCSolver()
        self.path = "uf20-01.cnf"

    def test_basic(self):
        book = Book("blablablabplqarblablablablalba")
        self.assertTrue(book._word_prob("hlla") > 0)
        
        #Test get_matrix_index from Table Class
        table = book._cut_sqrt_table(2,2,6)
        index = table._get_matrix_ind((16,2),10)
        self.assertTrue(index == (5,0))
        
        tables = book.find_tables(["pqr"])
        self.assertTrue(len(tables) > 0)
        self.assertTrue(tables[0].get_area() > 0 and tables[0].get_area() < 10)
        self.assertTrue(tables[0].get_area() ==  3)
    
    def test_read_file(self):
        
        currentLevel = 18
        (numVar, numClauses, clauses) = np.parseNPFunction(path = self.path)


        # Test parses the 0 and reads clauses and variables
        test = clauses.pop()
        self.assertTrue(len(test.intersection(set([0]))) == 0)
        clauses.add(test)
        self.assertTrue(numClauses > numVar)
        
    def test_min_clause(self):
        
        currentLevel = 18
        (numVar, numClauses, clauses) = np.parseNPFunction(path = self.path)

        minClauses = np.minimClause(clauses, currentLevel, newVar)
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
        
        minClauses = np.minimClause(clauses, currentLevel, newVar, False)
        for clause in minClauses:
            if newVar not in clause and -newVar not in clause:
                hasNewVariables = False
            if newVar in clause:
                hasPosSign = True
            elif -newVar in clause:
                hasNegSign = True
                
        self.assertTrue(not hasNewVariables)
        self.assertTrue(hasPosSign and hasNegSign)

        minClause = minimClause(clauses,level=8,newVariable=9)
    
    def test_update_state(self):
        
        
        #Based on minClause = 
        #{frozenset({9, -5, -1}),
        # frozenset({-8, 4, 7}), 
        #frozenset({9, 6, 1}),
        # frozenset({-8, 4, -9}),
        # frozenset({1, -5, -9})}
        #Value 100000001 only var1 at 1 and newVar=9 positive is not blocked by any
        #Value 000000001 all negative included newVar=-9 is blocked by {1, -5, -9}
        # So status should be 10b which is 2
        status2 = updateState(clauses, idValue=1, status=3, level=8, newVariable=9)
        self.assertTrue(status2==2)
        #Value 100000000 only newVar=9 positive is blocked by {9, -5, -1}
        #Value 000000000 all negative included newVar=-9 is not blocked by any
        # So status should be 01b which is 1
        status1 = updateState(clauses, idValue=0, status=3, level=8, newVariable=9)
        self.assertTrue(status1==1)